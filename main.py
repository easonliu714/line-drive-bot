import os
import json
import time
import logging
import shutil
import sys
import re
from flask import Flask, request, abort

from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, TextMessage, ImageMessage, FileMessage

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

# -------------------------------------------------
# 基本設定與日誌
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# -------------------------------------------------
# 安全讀取環境變數
# -------------------------------------------------

def get_env_var(key, default=None, required=True):
    val = os.environ.get(key, default)
    if required and not val:
        logger.critical(f"❌ 嚴重錯誤：找不到環境變數 '{key}'。請至 Cloud Run 設定變數。")
        return ""
    return str(val).strip()

LINE_TOKEN = get_env_var("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = get_env_var("LINE_CHANNEL_SECRET")
GEMINI_KEY = get_env_var("GEMINI_API_KEY")
GDRIVE_FOLDER_ID = get_env_var("GDRIVE_FOLDER_ID")

CLIENT_ID = get_env_var("GOOGLE_CLIENT_ID")
CLIENT_SECRET = get_env_var("GOOGLE_CLIENT_SECRET")
REFRESH_TOKEN = get_env_var("GOOGLE_REFRESH_TOKEN")

# -------------------------------------------------
# 模型設定 (可在此調整模型名稱)
# -------------------------------------------------
# 根據你的測試，使用能成功運作的名稱
# 常見有效名稱：gemini-1.5-flash, gemini-1.5-flash-latest, gemini-pro
MODEL_NAME = "gemini-1.5-flash-latest" 

if LINE_TOKEN and LINE_SECRET:
    line_bot_api = LineBotApi(LINE_TOKEN)
    parser = WebhookParser(LINE_SECRET)
else:
    logger.warning("⚠️ LINE Bot 設定不完整。")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    try:
        gemini_model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        logger.error(f"模型設定失敗，請確認名稱: {e}")

# -------------------------------------------------
# Session Manager
# -------------------------------------------------
user_sessions = {}

def start_session(user_id, context_name):
    user_sessions[user_id] = {
        'active': True,
        'context': context_name,
        'texts': [],
        'files': []
    }
    logger.info(f"User {user_id} started session: {context_name}")

def get_session(user_id):
    return user_sessions.get(user_id)

def add_to_session(user_id, text=None, file_path=None):
    if user_id not in user_sessions: return False
    if text: user_sessions[user_id]['texts'].append(text)
    if file_path: user_sessions[user_id]['files'].append(file_path)
    return True

def end_session(user_id):
    if user_id in user_sessions:
        return user_sessions.pop(user_id)
    return None

# -------------------------------------------------
# Google Services
# -------------------------------------------------

def get_drive_service():
    if not REFRESH_TOKEN or not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("OAuth Config Missing")
    creds = Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    return build("drive", "v3", credentials=creds)

def retry(func, retries=3, delay=2):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            logger.error(f"Retry {i+1} failed: {e}")
            time.sleep(delay)
    raise RuntimeError("All retries failed")

def clean_json_text(text: str) -> str:
    """清理 JSON 字串，移除 Markdown 與多餘文字"""
    text = text.strip()
    # 移除 ```json 包裹
    if "```" in text:
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text

def analyze_batch_content(context_name: str, texts: list, file_paths: list) -> dict:
    logger.info("呼叫 Gemini 進行批次分析...")
    combined_text = "\n".join(texts)
    has_images = len(file_paths) > 0
    
    # Prompt 優化：加入 "Output raw JSON only" 防止 AI 多嘴
    base_prompt = f"""
    Role: Professional Digital Archivist.
    Context: "{context_name}".
    Task: Analyze the provided content (text/images/links) and extract structured data.
    
    **CRITICAL OUTPUT RULE:** - Output ONLY valid JSON. 
    - Do NOT add any conversational text (e.g., "Here is the JSON", "I have analyzed...").
    - Do NOT use Markdown code blocks if possible, just the raw JSON string.

    Analysis Rules:
    1. **Summary**: Summarize the conversation or content.
    2. **Tags**: Extract keywords (Person, Action, Date, Place, Item).
    """

    if has_images:
        base_prompt += """
    3. **Images (OCR)**: Extract ALL visible text (Date, Amount, Shop Name). Describe the scene if no text.
        """
    else:
        base_prompt += """
    3. **Text/Links**: 
       - If URL exists, categorize it (e.g., 'Reference Link').
       - DO NOT hallucinate receipt/invoice details if no image is provided.
        """

    base_prompt += f"""
    **JSON Structure:**
    {{
      "source": "{context_name}",
      "category": "Category Name",
      "summary": "Detailed summary here...",
      "tags": ["tag1", "tag2"]
    }}
    """
    
    content_parts = [base_prompt]
    if combined_text: content_parts.append(f"\nContent:\n{combined_text}")
    
    for path in file_paths:
        try:
            mime_type = "image/jpeg"
            with open(path, "rb") as f:
                image_data = f.read()
            content_parts.append({"mime_type": mime_type, "data": image_data})
        except Exception as e:
            logger.error(f"讀圖失敗: {e}")

    # 初始化變數避免 UnboundLocalError
    response = None

    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = gemini_model.generate_content(
            content_parts, 
            safety_settings=safety_settings
        )
        
        if not response.parts:
            logger.warning("Gemini 回傳空內容 (Safety Filter)")
            return {"source": context_name, "category": "未分類", "summary": "內容被安全性阻擋", "tags": []}

        # Debug: 印出原始回應
        logger.info(f"Gemini Raw Response: {response.text}")
        
        return json.loads(clean_json_text(response.text))

    except Exception as e:
        logger.error(f"Gemini 分析/解析失敗: {e}")
        # 修正後的錯誤處理邏輯
        if response and hasattr(response, 'text') and response.text:
             return {
                 "source": context_name, 
                 "category": "格式錯誤", 
                 "summary": f"AI 回傳了非 JSON 格式，請檢查 Log。", 
                 "tags": ["error"]
             }
        return {"source": context_name, "category": "系統錯誤", "summary": "分析過程發生例外", "tags": ["error"]}

def get_or_create_folder(service, parent_id: str, folder_name: str) -> str:
    if not parent_id: raise ValueError("Parent ID is empty")
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])
    if files: return files[0]["id"]
    
    folder_metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    return folder["id"]

def get_target_folder_id(service, root_id, source_name, category_name):
    source_folder_id = get_or_create_folder(service, root_id, source_name)
    category_folder_id = get_or_create_folder(service, source_folder_id, category_name)
    return category_folder_id

def upload_file_to_drive(service, local_path: str, filename: str, folder_id: str, description: str = ""):
    logger.info(f"上傳: {filename} (Desc長度: {len(description)})")
    media = MediaFileUpload(local_path, resumable=True)
    file_metadata = {
        "name": filename, 
        "parents": [folder_id], 
        "description": description
    }
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()

# -------------------------------------------------
# Flask Routes
# -------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        events = parser.parse(body, signature)
    except Exception:
        abort(400)

    for event in events:
        if isinstance(event, MessageEvent):
            handle_message(event)

    return "OK", 200

def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    
    if isinstance(event.message, TextMessage):
        text = event.message.text.strip()
        
        # 指令：開始
        if text.startswith("開始") or text.lower().startswith("start"):
            parts = text.split(" ", 1)
            context = parts[1] if len(parts) > 1 else "未命名對話"
            start_session(user_id, context)
            line_bot_api.reply_message(event.reply_token, TextMessage(text=f"【錄製模式開啟】\n情境：「{context}」\n請轉傳訊息，完成後輸入「結束」。"))
            return

        # 指令：結束
        elif text == "結束" or text.lower() == "end":
            session = end_session(user_id)
            if not session:
                line_bot_api.reply_message(event.reply_token, TextMessage(text="目前沒有進行中的錄製。"))
                return
            
            line_bot_api.reply_message(event.reply_token, TextMessage(text=f"【處理中】\n共 {len(session['texts'])} 則訊息、{len(session['files'])} 個檔案..."))
            
            try:
                service = get_drive_service()
                ai_result = retry(lambda: analyze_batch_content(session['context'], session['texts'], session['files']))
                folder_id = get_target_folder_id(service, GDRIVE_FOLDER_ID, ai_result.get("source"), ai_result.get("category", "雜項"))
                
                full_description = f"AI摘要: {ai_result['summary']}\n\n標籤: {', '.join(ai_result.get('tags', []))}"

                if session['texts']:
                    log_filename = f"chat_batch_{int(time.time())}.txt"
                    with open(log_filename, "w", encoding="utf-8") as f:
                        f.write(f"情境：{session['context']}\n")
                        f.write(f"摘要：{ai_result['summary']}\n")
                        f.write(f"標籤：{', '.join(ai_result.get('tags', []))}\n")
                        f.write("-" * 20 + "\n")
                        f.write("\n".join(session['texts']))
                    
                    upload_file_to_drive(service, log_filename, log_filename, folder_id, description=full_description)
                    os.remove(log_filename)
                
                for fp in session['files']:
                    upload_file_to_drive(service, fp, os.path.basename(fp), folder_id, description=full_description)
                    os.remove(fp)

                line_bot_api.push_message(user_id, TextMessage(text=f"✅ 完成！\n分類：{ai_result.get('category')}\n重點：{ai_result.get('summary')[:100]}..."))

            except Exception as e:
                logger.error(f"批次失敗: {e}")
                line_bot_api.push_message(user_id, TextMessage(text="❌ 處理失敗，請檢查系統紀錄。"))
            return

    session = get_session(user_id)
    if session and session['active']:
        saved_path = None
        txt_content = None
        
        if isinstance(event.message, (ImageMessage, FileMessage)):
            msg_id = event.message.id
            ext = "jpg" if isinstance(event.message, ImageMessage) else "dat"
            saved_path = f"/tmp/batch_{msg_id}.{ext}"
            with open(saved_path, "wb") as f:
                for chunk in line_bot_api.get_message_content(msg_id).iter_content():
                    f.write(chunk)
        
        if isinstance(event.message, TextMessage):
            txt_content = event.message.text

        add_to_session(user_id, text=txt_content, file_path=saved_path)
    else:
        if isinstance(event.message, TextMessage):
             line_bot_api.reply_message(event.reply_token, TextMessage(text="請輸入「開始 [名稱]」來啟動智能歸檔模式。"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
