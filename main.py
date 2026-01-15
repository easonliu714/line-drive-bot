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
from google.oauth2 import service_account

# -------------------------------------------------
# 基本設定
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 環境變數（由 Cloud Run / Secret 注入）
# 使用 .strip() 去除可能誤貼的換行符號或空白
LINE_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"].strip()
GEMINI_KEY = os.environ["GEMINI_API_KEY"].strip()
GDRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()

line_bot_api = LineBotApi(LINE_TOKEN)
parser = WebhookParser(os.environ.get("LINE_CHANNEL_SECRET", ""))

genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel("gemini-flash-latest")

# -------------------------------------------------
# Google Drive（使用 Cloud Run 預設 Service Account）
# -------------------------------------------------

drive_service = build("drive", "v3")

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
        # 送出請求
        response = gemini_model.generate_content(prompt)
        
        # 1. 檢查是否有內容 (避免 Safety Filter 擋住導致無內容)
        if not response.parts:
            logger.warning("Gemini response blocked or empty.")
            return {"summary": text[:50], "tags": ["uncategorized"]}

        # 2. 清理 Markdown 格式 (去除 ```json 和 ```)
        content = response.text.strip()
        if content.startswith("```"):
            # 去掉開頭的 ```json 或 ```
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)
        
        # 3. 解析 JSON
        return json.loads(content)

    except Exception as e:
        logger.error(f"Gemini processing failed: {e}")
        # 萬一還是失敗，回傳一個預設值，確保程式不會 Crash
        return {"summary": text[:50], "tags": ["error"]}

def get_or_create_folder(parent_id: str, folder_name: str) -> str:
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = drive_service.files().create(body=folder_metadata, fields="id").execute()
    return folder["id"]


def upload_file_to_drive(local_path: str, filename: str, folder_id: str):
    media = MediaFileUpload(local_path, resumable=True)
    file_metadata = {"name": filename, "parents": [folder_id]}
    drive_service.files().create(body=file_metadata, media_body=media).execute()


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


# -------------------------------------------------
# LINE Message Handling
# -------------------------------------------------

def handle_message(event: MessageEvent):
    try:
        if isinstance(event.message, TextMessage):
            handle_text(event)

        elif isinstance(event.message, ImageMessage):
            handle_media(event, "image")

        elif isinstance(event.message, FileMessage):
            handle_media(event, "file")

    except Exception as e:
        logger.exception(f"Message handling failed: {e}")


def handle_text(event: MessageEvent):
    text = event.message.text
    result = retry(lambda: gemini_summarize_and_tag(text))

    folder_id = get_or_create_folder(GDRIVE_FOLDER_ID, result["tags"][0])
    filename = f"text_{int(time.time())}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(result["summary"] + "\n\n" + text)

    upload_file_to_drive(filename, filename, folder_id)
    os.remove(filename)


def handle_media(event: MessageEvent, media_type: str):
    message_id = event.message.id
    content = line_bot_api.get_message_content(message_id)

    filename = f"{media_type}_{int(time.time())}"
    with open(filename, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    # 簡易標籤（可之後加 OCR / metadata）
    tags = ["media"]
    folder_id = get_or_create_folder(GDRIVE_FOLDER_ID, tags[0])

    upload_file_to_drive(filename, filename, folder_id)
    os.remove(filename)


# -------------------------------------------------
# Entry
# -------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
