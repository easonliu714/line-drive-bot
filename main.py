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

# 使用 strip() 去除可能誤貼的換行符號或空白
LINE_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"].strip()
GEMINI_KEY = os.environ["GEMINI_API_KEY"].strip()
GDRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()

# Line Bot 設定 (注意：v2 寫法)
line_bot_api = LineBotApi(LINE_TOKEN)
parser = WebhookParser(os.environ.get("LINE_CHANNEL_SECRET", "").strip())

# Gemini 設定
genai.configure(api_key=GEMINI_KEY)
# 使用更穩定的模型名稱，若 flash-latest 有問題可改用 gemini-pro
gemini_model = genai.GenerativeModel("gemini-flash-latest")

# -------------------------------------------------
# Google Drive
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
    logger.info("呼叫 Gemini 進行摘要與標籤分類...")
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
            logger.warning("Gemini 回傳內容被阻擋或為空，使用預設值。")
            return {"summary": text[:50], "tags": ["未分類"]}

        content = response.text.strip()
        # 去除 Markdown 格式
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)
        
        result = json.loads(content)
        logger.info(f"Gemini 回傳成功: {result}")
        return result

    except Exception as e:
        logger.error(f"Gemini 處理失敗: {e}")
        return {"summary": text[:50], "tags": ["處理錯誤"]}


def get_or_create_folder(parent_id: str, folder_name: str) -> str:
    logger.info(f"正在搜尋或建立資料夾: {folder_name} (父目錄: {parent_id})")
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        logger.info(f"找到既有資料夾 ID: {files[0]['id']}")
        return files[0]["id"]

    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = drive_service.files().create(body=folder_metadata, fields="id").execute()
    logger.info(f"建立新資料夾 ID: {folder['id']}")
    return folder["id"]


def upload_file_to_drive(local_path: str, filename: str, folder_id: str):
    logger.info(f"開始上傳檔案: {filename} 到資料夾 ID: {folder_id}")
    try:
        # 明確指定 mimetype 為 text/plain 避免誤判
        media = MediaFileUpload(local_path, mimetype='text/plain', resumable=True)
        file_metadata = {"name": filename, "parents": [folder_id]}
        
        file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        logger.info(f"檔案上傳成功！檔案 ID: {file.get('id')}")
    except Exception as e:
        logger.error(f"檔案上傳失敗: {e}")
        raise e


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
        logger.exception(f"處理訊息時發生嚴重錯誤: {e}")


def handle_text(event: MessageEvent):
    text = event.message.text
    logger.info(f"收到文字訊息: {text[:20]}...")
    
    # 1. 取得摘要與標籤
    result = retry(lambda: gemini_summarize_and_tag(text))

    # 2. 處理資料夾
    folder_id = get_or_create_folder(GDRIVE_FOLDER_ID, result["tags"][0])
    
    # 3. 建立本地檔案
    filename = f"text_{int(time.time())}.txt"
    logger.info(f"正在建立本地檔案: {filename}")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"摘要：{result['summary']}\n\n原始內容：\n{text}")
        
        # 檢查檔案是否真的存在且有內容
        file_size = os.path.getsize(filename)
        logger.info(f"本地檔案建立完成，大小: {file_size} bytes")

        # 4. 上傳
        upload_file_to_drive(filename, filename, folder_id)

    except Exception as e:
        logger.error(f"檔案處理過程錯誤: {e}")
        raise e
    finally:
        # 清理
        if os.path.exists(filename):
            os.remove(filename)
            logger.info("本地暫存檔已刪除")


def handle_media(event: MessageEvent, media_type: str):
    message_id = event.message.id
    logger.info(f"收到媒體訊息 ID: {message_id}, 類型: {media_type}")
    
    content = line_bot_api.get_message_content(message_id)

    filename = f"{media_type}_{int(time.time())}"
    with open(filename, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    tags = ["媒體檔案"]
    folder_id = get_or_create_folder(GDRIVE_FOLDER_ID, tags[0])

    upload_file_to_drive(filename, filename, folder_id)
    os.remove(filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
