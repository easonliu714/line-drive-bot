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
from google.oauth2.credentials import Credentials

# -------------------------------------------------
# 基本設定
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 去除雜訊
LINE_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"].strip()
GEMINI_KEY = os.environ["GEMINI_API_KEY"].strip()
GDRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()

# OAuth 2.0 設定
CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"].strip()
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"].strip()
REFRESH_TOKEN = os.environ["GOOGLE_REFRESH_TOKEN"].strip()

line_bot_api = LineBotApi(LINE_TOKEN)
parser = WebhookParser(os.environ.get("LINE_CHANNEL_SECRET", "").strip())

genai.configure(api_key=GEMINI_KEY)
# 使用具備視覺能力的模型
gemini_model = genai.GenerativeModel("gemini-flash-latest")

# -------------------------------------------------
# Google Drive (使用 OAuth Refresh Token)
# -------------------------------------------------

def get_drive_service():
    creds = Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    return build("drive", "v3", credentials=creds)

# -------------------------------------------------
# 工具函式：Gemini AI 分析
# -------------------------------------------------

def retry(func, retries=3, delay=2):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            logger.error(f"Retry {i+1} failed: {e}")
            time.sleep(delay)
    raise RuntimeError("All retries failed")

def clean_json_text(text: str) -> str:
    """清理 Markdown 格式，只保留 JSON"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉第一行 ```json 和最後一行 ```
        if lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].startswith("```"): lines = lines[:-1]
        text = "\n".join(lines)
    return text

def analyze_content(text_content: str = None, image_data: bytes = None, mime_type: str = None) -> dict:
    """
    整合文字與圖片分析。
    回傳格式：{
        "source": "對話對象(人名/群組)",
        "category": "類別(工作/旅遊/發票...)",
        "summary": "摘要",
        "ocr_text": "圖片中的文字(如果是圖片)",
        "tags": ["tag1", "tag2"]
    }
    """
    logger.info("呼叫 Gemini 進行多模態分析...")
    
    # 建構 Prompt，教導 AI 如何分辨「對話對象」
    base_prompt = """
    你是一個專業的數位歸檔秘書。請分析使用者的輸入內容（文字或圖片）。
    請回傳 **純 JSON 格式**，不要有其他廢話。JSON 需包含以下欄位：
    {
      "source": "推測的對話對象或來源（例如：'客戶王大明', '老婆', '公司群組'）。如果無法判斷或只是單純檔案，請填 '未分類來源'",
      "category": "內容類別（例如：會議記錄, 旅遊行程, 報價單, 雜記事, 梗圖, 餐廳資訊）",
      "summary": "內容摘要（如果是圖片，請描述圖片內容）",
      "ocr_text": "如果是圖片，請盡量辨識圖中所有文字；如果是純文字則留空",
      "tags": ["標籤1", "標籤2", "關鍵字"]
    }

    **分析規則：**
    1. 如果是文字對話紀錄（例如包含 'A:', 'B:', 時間戳記），請從中提取人名作為 'source'。
    2. 如果是圖片，請根據圖片內容（如名片、Email截圖、LINE對話截圖）嘗試推斷 'source'，若看不出來則歸類為 '圖片歸檔'。
    3. 'ocr_text' 非常重要，請將圖片內可讀的文字轉出，方便搜尋。
    """

    content_parts = [base_prompt]
    
    if text_content:
        content_parts.append(f"\n文字內容：\n{text_content}")
    
    if image_data:
        content_parts.append({
            "mime_type": mime_type,
            "data": image_data
        })

    try:
        response = gemini_model.generate_content(content_parts)
        if not response.parts:
            return {"source": "未分類來源", "category": "未分類", "summary": "AI 無法讀取內容", "tags": [], "ocr_text": ""}
        
        json_str = clean_json_text(response.text)
        return json.loads(json_str)

    except Exception as e:
        logger.error(f"Gemini 分析失敗: {e}")
        # 回傳 fallback 結構以免程式崩潰
        return {
            "source": "處理錯誤",
            "category": "錯誤",
            "summary": f"分析失敗: {text_content[:20] if text_content else 'Media'}",
            "tags": ["error"],
            "ocr_text": ""
        }

# -------------------------------------------------
# Google Drive 資料夾邏輯 (三層結構)
# -------------------------------------------------

def get_target_folder_id(service, root_id, source_name, category_name):
    """
    建立路徑： Root -> Source (對象) -> Category (類別)
    回傳最後一層 Category 的 Folder ID
    """
    # 1. 處理第一層：對象 (Source)
    source_folder_id = get_or_create_folder(service, root_id, source_name)
    
    # 2. 處理第二層：類別 (Category)
    category_folder_id = get_or_create_folder(service, source_folder_id, category_name)
    
    return category_folder_id

def get_or_create_folder(service, parent_id: str, folder_name: str) -> str:
    # 搜尋是否已存在
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    # 不存在則建立
    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    logger.info(f"建立資料夾: {folder_name} (ID: {folder['id']})")
    return folder["id"]

def upload_file_to_drive(service, local_path: str, filename: str, folder_id: str, description: str = ""):
    logger.info(f"上傳: {filename} -> Folder: {folder_id}")
    # description 欄位可以用來放 OCR 文字或摘要，讓 Drive 搜尋更強大
    media = MediaFileUpload(local_path, resumable=True)
    file_metadata = {
        "name": filename, 
        "parents": [folder_id],
        "description": description 
    }
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()

# -------------------------------------------------
# Flask & Message Handling
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
            # 每次處理都重新取得 Drive Service (確保 Token 刷新)
            service = get_drive_service()
            handle_message(event, service)

    return "OK", 200

def handle_message(event: MessageEvent, service):
    try:
        ai_result = {}
        local_files = [] # 儲存要上傳的檔案路徑列表 (路徑, 檔名)

        # ---------------------------------------
        # 情境 A: 純文字 (可能是對話紀錄)
        # ---------------------------------------
        if isinstance(event.message, TextMessage):
            text = event.message.text
            # 呼叫 AI 分析
            ai_result = retry(lambda: analyze_content(text_content=text))
            
            # 建立文字檔
            filename = f"chat_{int(time.time())}.txt"
            content_to_save = f"摘要：{ai_result['summary']}\n\n原始內容：\n{text}"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content_to_save)
            local_files.append((filename, filename))

        # ---------------------------------------
        # 情境 B: 圖片 (需要 OCR 與視覺分析)
        # ---------------------------------------
        elif isinstance(event.message, ImageMessage):
            msg_id = event.message.id
            content = line_bot_api.get_message_content(msg_id)
            image_data = content.content # 這是 bytes

            # 呼叫 AI 分析 (傳入圖片 bytes)
            ai_result = retry(lambda: analyze_content(image_data=image_data, mime_type="image/jpeg"))

            # 儲存圖片
            img_filename = f"image_{int(time.time())}.jpg"
            with open(img_filename, "wb") as f:
                f.write(image_data)
            local_files.append((img_filename, img_filename))

            # 如果有 OCR 文字，額外存一個文字檔方便閱讀 (選擇性)
            if ai_result.get("ocr_text"):
                txt_filename = f"ocr_{int(time.time())}.txt"
                with open(txt_filename, "w", encoding="utf-8") as f:
                    f.write(f"圖片摘要：{ai_result['summary']}\n\nOCR 辨識文字：\n{ai_result['ocr_text']}")
                local_files.append((txt_filename, txt_filename))

        # ---------------------------------------
        # 共同後續：建立資料夾並上傳
        # ---------------------------------------
        if ai_result:
            # 1. 取得目標資料夾 (Root -> 對象 -> 類別)
            folder_id = get_target_folder_id(
                service, 
                GDRIVE_FOLDER_ID, 
                ai_result.get("source", "未分類來源"), 
                ai_result.get("category", "雜項")
            )

            # 2. 上傳所有檔案
            for path, name in local_files:
                # 將 OCR 結果放入檔案描述 (Description) 讓 Drive 搜尋得到
                desc = f"摘要: {ai_result.get('summary', '')}\n標籤: {','.join(ai_result.get('tags', []))}"
                upload_file_to_drive(service, path, name, folder_id, description=desc)
                
                # 刪除本地暫存
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info("處理完成！")

    except Exception as e:
        logger.exception(f"處理訊息失敗: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
