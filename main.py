import os
import json
import time
import logging
import shutil
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
gemini_model = genai.GenerativeModel("gemini-flash-latest")

# -------------------------------------------------
# Session Manager (批次處理邏輯)
# -------------------------------------------------

# 用來暫存使用者的對話狀態與緩衝區
# 結構: { 'user_id': { 'active': True, 'context': '與客戶A', 'texts': [], 'files': [] } }
user_sessions = {}

def get_session(user_id):
    return user_sessions.get(user_id)

def start_session(user_id, context_name):
    user_sessions[user_id] = {
        'active': True,
        'context': context_name, # 使用者指定的情境 (如：與老婆的對話)
        'texts': [],
        'files': []
    }
    logger.info(f"User {user_id} started session: {context_name}")

def add_to_session(user_id, text=None, file_path=None):
    if user_id not in user_sessions:
        return False
    
    if text:
        user_sessions[user_id]['texts'].append(text)
    if file_path:
        user_sessions[user_id]['files'].append(file_path)
    return True

def end_session(user_id):
    if user_id in user_sessions:
        session_data = user_sessions.pop(user_id)
        return session_data
    return None

# -------------------------------------------------
# Google Drive & Gemini
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
    """
    批次分析：一次把所有文字和圖片丟給 Gemini
    """
    logger.info("呼叫 Gemini 進行批次分析...")
    
    # 組合所有文字訊息
    combined_text = "\n".join(texts)
    
    # 建構 Prompt
    base_prompt = f"""
    你是一個專業的數位歸檔秘書。這裡有一組來自 LINE 的轉傳訊息。
    使用者已指定這組對話的情境/對象為：「{context_name}」。

    請依據這個情境與提供的內容（文字與圖片），回傳一個 JSON：
    {{
      "source": "{context_name}",
      "category": "內容類別（例如：會議記錄, 旅遊行程, 報價單, 閒聊, 待辦事項）",
      "summary": "這整組對話或檔案的綜合摘要",
      "tags": ["標籤1", "標籤2", "關鍵字"]
    }}
    
    注意：
    1. 'source' 請直接使用使用者提供的「{context_name}」，除非內容明顯衝突。
    2. 請歸納出整組對話的重點，不要逐字翻譯。
    """

    content_parts = [base_prompt]
    
    if combined_text:
        content_parts.append(f"\n對話文字內容：\n{combined_text}")
    
    # 加入圖片 (Gemini 支援多張圖同時分析)
    for path in file_paths:
        if path.endswith(('.jpg', '.jpeg', '.png')):
            try:
                mime_type = "image/jpeg" # 簡化處理，假設都是 jpg/png
                with open(path, "rb") as f:
                    image_data = f.read()
                content_parts.append({
                    "mime_type": mime_type,
                    "data": image_data
                })
            except Exception as e:
                logger.error(f"讀取圖片失敗 {path}: {e}")

    try:
        response = gemini_model.generate_content(content_parts)
        if not response.parts:
            return {"source": context_name, "category": "未分類", "summary": "AI 無法讀取內容", "tags": []}
        
        json_str = clean_json_text(response.text)
        return json.loads(json_str)

    except Exception as e:
        logger.error(f"Gemini 分析失敗: {e}")
        return {
            "source": context_name,
            "category": "錯誤",
            "summary": "分析失敗",
            "tags": ["error"]
        }

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
    logger.info(f"上傳: {filename}")
    media = MediaFileUpload(local_path, resumable=True)
    file_metadata = {"name": filename, "parents": [folder_id], "description": description}
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
            handle_message(event)

    return "OK", 200

def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    service = get_drive_service() # 準備 Drive 連線
    
    # -------------------------------------------------
    # 1. 處理文字指令 (開始/結束)
    # -------------------------------------------------
    if isinstance(event.message, TextMessage):
        text = event.message.text.strip()
        
        # 指令：開始 [情境]
        if text.startswith("開始") or text.startswith("Start"):
            # 解析情境名稱，例如 "開始 與客戶開會" -> context="與客戶開會"
            parts = text.split(" ", 1)
            context = parts[1] if len(parts) > 1 else "未命名對話"
            
            start_session(user_id, context)
            line_bot_api.reply_message(event.reply_token, TextMessage(text=f"【錄製模式開啟】\n設定情境為：「{context}」\n請開始轉傳訊息或圖片，完成後請輸入「結束」。"))
            return

        # 指令：結束
        elif text == "結束" or text == "End":
            session = end_session(user_id)
            if not session:
                line_bot_api.reply_message(event.reply_token, TextMessage(text="目前沒有進行中的錄製模式。"))
                return
            
            line_bot_api.reply_message(event.reply_token, TextMessage(text=f"【錄製結束】\n收到 {len(session['texts'])} 則訊息、{len(session['files'])} 個檔案。\n正在進行 AI 分析與歸檔，請稍候..."))
            
            # --- 執行批次處理 ---
            try:
                # 1. Gemini 分析
                ai_result = retry(lambda: analyze_batch_content(session['context'], session['texts'], session['files']))
                
                # 2. 建立資料夾
                folder_id = get_target_folder_id(
                    service, 
                    GDRIVE_FOLDER_ID, 
                    ai_result.get("source", session['context']), 
                    ai_result.get("category", "雜項")
                )
                
                # 3. 彙整文字檔 (Chat Log)
                if session['texts']:
                    log_filename = f"chat_batch_{int(time.time())}.txt"
                    with open(log_filename, "w", encoding="utf-8") as f:
                        f.write(f"情境：{session['context']}\n")
                        f.write(f"AI 摘要：{ai_result['summary']}\n")
                        f.write(f"標籤：{','.join(ai_result.get('tags', []))}\n")
                        f.write("-" * 20 + "\n")
                        f.write("\n".join(session['texts']))
                    
                    desc = f"摘要: {ai_result['summary']}"
                    upload_file_to_drive(service, log_filename, log_filename, folder_id, description=desc)
                    os.remove(log_filename)
                
                # 4. 上傳圖片檔案
                for file_path in session['files']:
                    upload_file_to_drive(service, file_path, os.path.basename(file_path), folder_id, description="批次上傳媒體")
                    os.remove(file_path)

                line_bot_api.push_message(user_id, TextMessage(text=f"✅ 歸檔完成！\n分類：{ai_result.get('category')}\n摘要：{ai_result.get('summary')}"))

            except Exception as e:
                logger.error(f"批次處理失敗: {e}")
                line_bot_api.push_message(user_id, TextMessage(text="❌ 處理過程中發生錯誤，請檢查系統紀錄。"))
            
            return

    # -------------------------------------------------
    # 2. 處理內容 (收集模式 vs 單則模式)
    # -------------------------------------------------
    
    session = get_session(user_id)
    
    # 暫存變數
    saved_file_path = None
    text_content = None

    # 下載檔案 (如果是圖片/檔案)
    if isinstance(event.message, (ImageMessage, FileMessage)):
        msg_id = event.message.id
        content = line_bot_api.get_message_content(msg_id)
        
        ext = "jpg" if isinstance(event.message, ImageMessage) else "dat"
        saved_file_path = f"batch_{msg_id}.{ext}"
        
        with open(saved_file_path, "wb") as f:
            for chunk in content.iter_content():
                f.write(chunk)

    # 取得文字
    if isinstance(event.message, TextMessage):
        text_content = event.message.text

    # --- 判斷邏輯 ---
    if session and session['active']:
        # [錄製模式中]：只儲存，不分析
        add_to_session(user_id, text=text_content, file_path=saved_file_path)
        # 不回覆訊息，以免打擾轉傳過程
    
    else:
        # [非錄製模式]：維持原本的單則處理邏輯 (Fallback)
        # 這裡為了簡化，你可以選擇「不處理」或是「警告使用者要先輸入開始」
        # 或是保留原本的單則分析邏輯 (如果你希望保留單傳功能)
        
        if saved_file_path: os.remove(saved_file_path) # 清理未使用的檔案
        # 提示使用者使用新功能
        if isinstance(event.message, TextMessage):
             line_bot_api.reply_message(event.reply_token, TextMessage(text="請輸入「開始 [對象]」進入批次歸檔模式，\n例如：「開始 與客戶A的討論」"))
