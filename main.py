import os
import json
import time
import logging
from flask import Flask, request, abort

from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, TextMessage, ImageMessage, FileMessage

import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials  # 改用這個

# -------------------------------------------------
# 基本設定
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 使用 strip() 去除雜訊
LINE_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"].strip()
GEMINI_KEY = os.environ["GEMINI_API_KEY"].strip()
GDRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()

# OAuth 2.0 設定 (取代原本的 Service Account)
CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"].strip()
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"].strip()
REFRESH_TOKEN = os.environ["GOOGLE_REFRESH_TOKEN"].strip()

line_bot_api = LineBotApi(LINE_TOKEN)
parser = WebhookParser(os.environ.get("LINE_CHANNEL_SECRET", "").strip())

genai.configure(api_key=GEMINI_KEY)
# 使用 gemini-pro 或 flash-latest
gemini_model = genai.GenerativeModel("gemini-flash-latest")

# -------------------------------------------------
# Google Drive (使用 OAuth Refresh Token)
# -------------------------------------------------

def get_drive_service():
    # 每次呼叫時建立憑證，確保 Token 自動刷新
    creds = Credentials(
        None, # 沒有 Access Token，會自動用 Refresh Token 換
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    return build("drive", "v3", credentials=creds)

# -------------------------------------------------
# 工具函式
# -------------------------------------------------

def retry(func, retries=3, delay=2):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            logger.error(f"Retry {i+1} failed: {e}")
            time.sleep(delay)
    raise RuntimeError("All retries failed")


def gemini_summarize_and_tag(text: str) -> dict:
    logger.info("呼叫 Gemini...")
    prompt = f"""
請用繁體中文回覆，格式為 JSON：
{{
  "summary": "一句話摘要",
  "tags": ["標籤1", "標籤2"]
}}

內容：
{text}
"""
    try:
        response = gemini_model.generate_content(prompt)
        
        if not response.parts:
            return {"summary": text[:50], "tags": ["未分類"]}

        content = response.text.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)
        
        return json.loads(content)

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {"summary": text[:50], "tags": ["error"]}


def get_or_create_folder(service, parent_id: str, folder_name: str) -> str:
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    return folder["id"]


def upload_file_to_drive(service, local_path: str, filename: str, folder_id: str):
    logger.info(f"上傳檔案中: {filename}")
    media = MediaFileUpload(local_path, mimetype='text/plain', resumable=True)
    file_metadata = {"name": filename, "parents": [folder_id]}
    
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()


# -------------------------------------------------
# Flask Routes & Line Handling
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
    try:
        # 每次處理訊息時取得連線 (避免 Token 過期問題)
        service = get_drive_service()

        if isinstance(event.message, TextMessage):
            text = event.message.text
            result = retry(lambda: gemini_summarize_and_tag(text))
            
            folder_id = get_or_create_folder(service, GDRIVE_FOLDER_ID, result["tags"][0])
            filename = f"text_{int(time.time())}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"摘要：{result['summary']}\n\n{text}")
            
            upload_file_to_drive(service, filename, filename, folder_id)
            os.remove(filename)

        elif isinstance(event.message, ImageMessage) or isinstance(event.message, FileMessage):
            media_type = "image" if isinstance(event.message, ImageMessage) else "file"
            message_id = event.message.id
            content = line_bot_api.get_message_content(message_id)

            filename = f"{media_type}_{int(time.time())}"
            with open(filename, "wb") as f:
                for chunk in content.iter_content():
                    f.write(chunk)

            folder_id = get_or_create_folder(service, GDRIVE_FOLDER_ID, "媒體檔案")
            upload_file_to_drive(service, filename, filename, folder_id)
            os.remove(filename)

    except Exception as e:
        logger.exception(f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
