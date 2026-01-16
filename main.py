import os
import json
import time
import logging
import shutil
import sys
from flask import Flask, request, abort

from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, TextMessage, ImageMessage, FileMessage

import google.generativeai as genai
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

if LINE_TOKEN and LINE_SECRET:
    line_bot_api = LineBotApi(LINE_TOKEN)
    parser = WebhookParser(LINE_SECRET)
else:
    logger.warning("⚠️ LINE Bot 設定不完整。")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    gemini_model = genai.GenerativeModel("gemini-flash-latest")

# -------------------------------------------------
# Session Manager (錄製模式)
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
# Google Services 工具函式
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
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        text = "\n".join(lines)
    return text

def analyze_batch_content(context_name: str, texts: list, file_paths: list) -> dict:
    logger.info("呼叫 Gemini 進行批次分析...")
    combined_text = "\n".join(texts)
    has_images = len(file_paths) > 0
    
    # -------------------------------------------------
    # Prompt 優化：加入「有無圖片」的邏輯判斷，防止幻覺
    # -------------------------------------------------
    
    # 基礎指令
    base_prompt = f"""
    你是一個專業的數位歸檔秘書。
    使用者指定的情境為：「{context_name}」。
    你收到了使用者轉傳的內容。

    **你的任務目標：**
    1. **分析內容**：根據提供的文字{ "與圖片" if has_images else "" }進行歸納。
    2. **標籤提取**：提取人、事、時、地、物作為標籤。
    """

    # 針對圖片的特殊指令 (只有在真的有圖片時才加入)
    if has_images:
        base_prompt += """
    3. **OCR 與視覺分析**：
       - 請仔細閱讀圖片中的文字。
       - 如果是收據、文件、名片，**請務必提取**日期、金額、店名。
       - 如果是活動照片，請描述畫面內容。
        """
    else:
        base_prompt += """
    3. **純文字處理**：
       - **嚴禁捏造**：既然沒有提供圖片，請不要幻想出任何圖片的分析結果（如收據、發票）。
       - **網址處理**：如果內容包含 URL (https://...)，請將其記錄為「參考連結」，不要試圖分析網頁內文（除非使用者有提供網頁截圖）。
        """

    # 結尾格式要求
    base_prompt += f"""
    請回傳 JSON 格式：
    {{
      "source": "{context_name}",
      "category": "精確類別",
      "summary": "重點摘要。{'包含圖片OCR結果' if has_images else '針對文字對話的總結' }。",
      "tags": ["關鍵字1", "關鍵字2"]
    }}
    注意：'source' 請優先使用「{context_name}」。
    """
    
    content_parts = [base_prompt]
    if combined_text: content_parts.append(f"\n文字與連結內容：\n{combined_text}")
    
    for path in file_paths:
        try:
            mime_type = "image/jpeg"
            with open(path, "rb") as f:
                image_data = f.read()
            content_parts.append({"mime_type": mime_type, "data": image_data})
        except Exception as e:
            logger.error(f"讀圖失敗: {e}")

    try:
        response = gemini_model.generate_content(content_parts)
        if not response.parts: return {"source": context_name, "category": "未分類", "summary": "AI 無法讀取內容", "tags": []}
        return json.loads(clean_json_text(response.text))
    except Exception as e:
        logger.error(f"Gemini 分析失敗: {e}")
        return {"source": context_name, "category": "錯誤", "summary": "分析失敗", "tags": ["error"]}

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
        "description": description  # 關鍵：將 OCR 結果寫入檔案說明
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
    
    # 處理文字指令
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
            
            line_bot_api.reply_message(event.reply_token, TextMessage(text=f"【處理中】\n共 {len(session['texts'])} 則訊息、{len(session['files'])} 個檔案...\n正在提取重點與 OCR..."))
            
            try:
                service = get_drive_service()
                # 1. AI 分析
                ai_result = retry(lambda: analyze_batch_content(session['context'], session['texts'], session['files']))
                folder_id = get_target_folder_id(service, GDRIVE_FOLDER_ID, ai_result.get("source"), ai_result.get("category", "雜項"))
                
                # 準備寫入 Description 的內容 (摘要 + 標籤)
                # 這會讓你在 Drive 搜尋時能搜到圖片內的文字
                full_description = f"AI摘要: {ai_result['summary']}\n\n標籤: {', '.join(ai_result.get('tags', []))}"

                # 2. 彙整文字檔
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
                
                # 3. 上傳圖片 (關鍵：寫入 Description)
                for fp in session['files']:
                    upload_file_to_drive(
                        service, 
                        fp, 
                        os.path.basename(fp), 
                        folder_id, 
                        description=full_description # 每一張圖片都會帶有這次分析的重點摘要
                    )
                    os.remove(fp)

                line_bot_api.push_message(user_id, TextMessage(text=f"✅ 完成！\n分類：{ai_result.get('category')}\n重點：{ai_result.get('summary')[:100]}..."))

            except Exception as e:
                logger.error(f"批次失敗: {e}")
                line_bot_api.push_message(user_id, TextMessage(text="❌ 處理失敗，請檢查系統紀錄。"))
            return

    # 錄製過程
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
        # 非錄製模式提示
        if isinstance(event.message, TextMessage):
             line_bot_api.reply_message(event.reply_token, TextMessage(text="請輸入「開始 [名稱]」來啟動智能歸檔模式。\n我會幫你做 OCR 與重點整理！"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
