import os
import json
import time
import logging
import shutil
import sys
import re
from datetime import datetime, timedelta
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
# 環境變數讀取
# -------------------------------------------------

def get_env_var(key, default=None, required=True):
    val = os.environ.get(key, default)
    if required and not val:
        logger.critical(f"❌ 嚴重錯誤：找不到環境變數 '{key}'。")
        return ""
    return str(val).strip()

LINE_TOKEN = get_env_var("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = get_env_var("LINE_CHANNEL_SECRET")
GEMINI_KEY = get_env_var("GEMINI_API_KEY")
GDRIVE_FOLDER_ID = get_env_var("GDRIVE_FOLDER_ID")

CLIENT_ID = get_env_var("GOOGLE_CLIENT_ID")
CLIENT_SECRET = get_env_var("GOOGLE_CLIENT_SECRET")
REFRESH_TOKEN = get_env_var("GOOGLE_REFRESH_TOKEN")

# 模型名稱設定
MODEL_NAME = "gemini-flash-latest"

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
        logger.error(f"模型設定失敗: {e}")

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
# Google Services (Drive & Calendar)
# -------------------------------------------------

def get_google_creds():
    if not REFRESH_TOKEN or not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("OAuth Config Missing")
    return Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )

def get_drive_service():
    return build("drive", "v3", credentials=get_google_creds())

def get_calendar_service():
    return build("calendar", "v3", credentials=get_google_creds())

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
    if "```" in text:
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text

# -------------------------------------------------
# AI Analysis Logic
# -------------------------------------------------

def analyze_batch_content(context_name: str, texts: list, file_paths: list) -> dict:
    logger.info("呼叫 Gemini 進行分析...")
    combined_text = "\n".join(texts)
    has_images = len(file_paths) > 0
    
    # 取得當前時間，讓 AI 知道「明天」是幾號
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S (Asia/Taipei)")

    # -------------------------------------------------
    # Prompt 設計：強調中文、行事曆提取、對話識別
    # -------------------------------------------------
    base_prompt = f"""
    You are a professional digital secretary.
    Current Time: {current_time}.
    User Context: "{context_name}".
    
    **MANDATORY RULES:**
    1. **Language**: OUTPUT MUST BE IN **TRADITIONAL CHINESE (繁體中文)**.
    2. **Format**: Output ONLY raw JSON. No markdown formatting if possible.
    
    **Task 1: Content Analysis**
    - Analyze the input text/images.
    - If the context implies a conversation (e.g., "{context_name}"), try to infer speakers based on content logic even if explicit names are missing.
    - If multiple speakers are detected, summarize their key points separately in the summary.

    **Task 2: Calendar Extraction**
    - Detect any actionable events, meetings, or deadlines.
    - **Crucial**: Convert relative dates (e.g., "tomorrow afternoon", "next Friday") into exact ISO 8601 timestamps based on 'Current Time'.
    - If no specific time is mentioned for a date, assume 09:00 for start and 10:00 for end.
    
    """

    if has_images:
        base_prompt += """
    **Task 3: OCR (Images)**
    - Extract Text: Date, Time, Amount, Location, Topic.
    - Merge this info with text analysis.
        """
    else:
        base_prompt += """
    **Task 3: Text Only**
    - Do not hallucinate image details.
        """

    base_prompt += f"""
    **JSON Output Schema:**
    {{
      "source": "{context_name}",
      "category": "Category (e.g., 會議, 旅遊, 財務, 待辦)",
      "summary": "Comprehensive summary in Traditional Chinese. Include who said what if applicable.",
      "tags": ["tag1", "tag2"],
      "calendar_events": [
        {{
            "summary": "Event Title",
            "start_time": "YYYY-MM-DDTHH:MM:SS",
            "end_time": "YYYY-MM-DDTHH:MM:SS",
            "location": "Location (optional)"
        }}
      ]
    }}
    * If no events are found, "calendar_events" should be an empty list [].
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

    # Safety Settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        response = gemini_model.generate_content(content_parts, safety_settings=safety_settings)
        if not response.parts:
            return {"source": context_name, "category": "未分類", "summary": "內容被阻擋", "tags": [], "calendar_events": []}
        
        logger.info(f"AI Raw Response: {response.text}")
        return json.loads(clean_json_text(response.text))

    except Exception as e:
        logger.error(f"AI Error: {e}")
        # Fallback if JSON fails
        return {
            "source": context_name,
            "category": "錯誤",
            "summary": "AI 分析失敗，請檢查 Log",
            "tags": ["error"],
            "calendar_events": []
        }

# -------------------------------------------------
# Drive & Calendar Operations
# -------------------------------------------------

def add_calendar_events(events_list):
    """將 AI 分析出的事件寫入 Google Calendar"""
    if not events_list:
        return 0
    
    try:
        service = get_calendar_service()
        count = 0
        for event in events_list:
            logger.info(f"Adding Event: {event}")
            # 建構事件物件
            evt_body = {
                'summary': event.get('summary', '未命名事件'),
                'location': event.get('location', ''),
                'description': '由 LINE Bot 自動建立',
                'start': {
                    'dateTime': event['start_time'],
                    'timeZone': 'Asia/Taipei',
                },
                'end': {
                    'dateTime': event['end_time'],
                    'timeZone': 'Asia/Taipei',
                },
            }
            service.events().insert(calendarId='primary', body=evt_body).execute()
            count += 1
        return count
    except Exception as e:
        logger.error(f"Calendar Insert Failed: {e}")
        return 0

def get_or_create_folder(service, parent_id: str, folder_name: str) -> str:
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
    media = MediaFileUpload(local_path, resumable=True)
    file_metadata = {"name": filename, "parents": [folder_id], "description": description}
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()

# -------------------------------------------------
# Flask Routes & Bot Logic
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
            line_bot_api.reply_message(event.reply_token, TextMessage(text=f"【錄製模式開啟】\n📝 情境：「{context}」\n💡 提示：若包含日期行程，將自動加入行事曆。\n💡 提示：若是多人對話，請盡量複製文字貼上以利辨識。"))
            return

        # 指令：結束
        elif text == "結束" or text.lower() == "end":
            session = end_session(user_id)
            if not session:
                line_bot_api.reply_message(event.reply_token, TextMessage(text="目前沒有進行中的錄製。"))
                return
            
            line_bot_api.reply_message(event.reply_token, TextMessage(text=f"🤖 處理中...\n訊息：{len(session['texts'])} 則\n檔案：{len(session['files'])} 個"))
            
            try:
                # 1. AI 分析
                ai_result = retry(lambda: analyze_batch_content(session['context'], session['texts'], session['files']))
                
                # 2. 處理 Drive 歸檔
                drive_service = get_drive_service()
                folder_id = get_target_folder_id(drive_service, GDRIVE_FOLDER_ID, ai_result.get("source"), ai_result.get("category", "雜項"))
                
                full_desc = f"AI摘要: {ai_result['summary']}\n標籤: {', '.join(ai_result.get('tags', []))}"

                # 儲存文字記錄
                if session['texts']:
                    log_filename = f"chat_batch_{int(time.time())}.txt"
                    with open(log_filename, "w", encoding="utf-8") as f:
                        f.write(f"情境：{session['context']}\n")
                        f.write(f"摘要：{ai_result['summary']}\n")
                        f.write("-" * 20 + "\n")
                        f.write("\n".join(session['texts']))
                    upload_file_to_drive(drive_service, log_filename, log_filename, folder_id, description=full_desc)
                    os.remove(log_filename)
                
                # 儲存圖片
                for fp in session['files']:
                    upload_file_to_drive(drive_service, fp, os.path.basename(fp), folder_id, description=full_desc)
                    os.remove(fp)

                # 3. 處理行事曆 (新功能)
                cal_count = 0
                if ai_result.get("calendar_events"):
                    cal_count = add_calendar_events(ai_result["calendar_events"])

                # 回覆結果
                reply_msg = f"✅ 歸檔完成！(分類：{ai_result.get('category')})\n\n摘要：\n{ai_result.get('summary')}"
                if cal_count > 0:
                    reply_msg += f"\n\n📅 已自動加入 {cal_count} 個行事曆行程！"
                
                line_bot_api.push_message(user_id, TextMessage(text=reply_msg))

            except Exception as e:
                logger.error(f"批次失敗: {e}")
                line_bot_api.push_message(user_id, TextMessage(text="❌ 處理失敗，請檢查系統紀錄。"))
            return

    # 錄製中
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
             line_bot_api.reply_message(event.reply_token, TextMessage(text="請輸入「開始 [名稱]」來啟動模式。\n例如：開始 君君和流星雨的對話"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
