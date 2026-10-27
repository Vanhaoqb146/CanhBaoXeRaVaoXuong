"""Compatibility helper: unchanged send_alert contract, R2 replaces ImgBB."""

import datetime
import json
import threading
import time
from pathlib import Path

import requests

from image_storage import R2Storage
from r2_settings import APP_DIR, StorageError

ZALO_API_BASE_URL = "https://bot-api.zaloplatforms.com"


def load_alert_config(config_path=None):
    path = Path(config_path) if config_path else APP_DIR / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise StorageError("Không đọc được config.json cho Zalo; kiểm tra file và cú pháp JSON.") from None
    if not isinstance(config, dict) or not isinstance(config.get("zalo_bots", []), list):
        raise StorageError("config.json phải có zalo_bots dạng danh sách.")
    # A token is part of the URL; refuse unknown endpoints instead of leaking it.
    base_url = config.get("zalo_api_base_url", ZALO_API_BASE_URL)
    if not isinstance(base_url, str) or base_url.rstrip("/") != ZALO_API_BASE_URL:
        raise StorageError("zalo_api_base_url phải là endpoint Zalo Bot chính thức.")
    return config


def valid_bot(bot):
    if not isinstance(bot, dict):
        return False
    token, chat_id = bot.get("bot_token"), bot.get("chat_id")
    return (
        isinstance(token, str)
        and bool(token.strip())
        and not token.startswith(("DAN_TOKEN", "<"))
        and all(character.isalnum() or character in ":_-" for character in token)
        and isinstance(chat_id, (str, int))
        and not isinstance(chat_id, bool)
        and bool(str(chat_id).strip())
    )


def _read_zalo_response(response, label):
    try:
        data = response.json()
    except ValueError:
        print(f"[{label}] Zalo trả phản hồi không phải JSON (HTTP {response.status_code}).")
        return False
    # Do not log response text/description: providers may echo tokens or signed URLs.
    if response.status_code != 200 or not isinstance(data, dict) or data.get("ok") is not True:
        print(f"[{label}] Zalo từ chối ảnh (HTTP {response.status_code}); kiểm tra bot/chat/quota.")
        return False
    print(f"[{label}] Gửi ảnh Zalo thành công.")
    return True


def _send_photo_by_url(bot_config, image_url, message_text, label="bot", max_attempts=3):
    token = bot_config["bot_token"]
    url = f"{ZALO_API_BASE_URL}/bot{token}/sendPhoto"
    now = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    payload = {
        "chat_id": str(bot_config["chat_id"]),
        "photo": image_url,
        "caption": f"{message_text}\n🕒 {now}",
    }
    if len(payload["caption"]) > 2000:
        print(f"[{label}] Caption vượt giới hạn 2000 ký tự; chưa gửi.")
        return False
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, json=payload, timeout=(5, 15), allow_redirects=False)
        except requests.exceptions.ConnectTimeout:
            if attempt < max_attempts:
                time.sleep(2)
                continue
            print(f"[{label}] Không kết nối được Zalo; đã hết số lần thử.")
            return False
        except requests.exceptions.RequestException:
            # A POST read timeout may mean delivery succeeded: don't duplicate the alert.
            print(f"[{label}] Lỗi mạng Zalo; chưa xác định đã giao tin, không tự gửi lại.")
            return False
        try:
            if response.status_code == 429 and attempt < max_attempts:
                try:
                    delay = min(30, max(2, int(response.headers.get("Retry-After", "2"))))
                except (TypeError, ValueError):
                    delay = 2
                time.sleep(delay)
                continue
            return _read_zalo_response(response, label)
        finally:
            response.close()
    return False


def upload_to_r2(image_path):
    """Return a temporary GET URL, or None; never print the signed URL."""
    try:
        uploaded = R2Storage.from_config().upload(image_path)
    except StorageError as error:
        print(f"R2: {error}")
        return None
    print("R2: upload ảnh thành công, đã tạo URL GET có hạn.")
    return uploaded.url


def send_alert_sync(
    image_local_path, message_text="🚨 CẢNH BÁO AN NINH", *, config=None, bots=None, max_attempts=3
):
    """Blocking variant for diagnostics. True means at least one bot confirmed ok."""
    try:
        config = load_alert_config() if config is None else config
        selected = config.get("zalo_bots", []) if bots is None else bots
        selected = [(index, bot) for index, bot in enumerate(selected, 1) if valid_bot(bot)]
        if not selected:
            print("Zalo: không có bot/chat hợp lệ; chưa upload hoặc gửi ảnh.")
            return False
        image_url = upload_to_r2(image_local_path)
        if not image_url:
            return False
        successes = sum(
            _send_photo_by_url(bot, image_url, message_text, f"bot {index}", max_attempts)
            for index, bot in selected
        )
        print(f"Kết quả Zalo: {successes}/{len(selected)} bot xác nhận thành công.")
        return successes > 0
    except StorageError as error:
        print(f"Cấu hình gửi ảnh: {error}")
    except Exception:
        print("Gửi ảnh thất bại do lỗi nội bộ; không ghi exception thô chứa thông tin riêng.")
    return False


def _send_alert_sync(image_local_path, message_text):
    return send_alert_sync(image_local_path, message_text)


def send_alert(image_local_path, message_text="🚨 CẢNH BÁO AN NINH"):
    """Preserve background dispatch semantics of the existing main.py."""
    thread = threading.Thread(target=_send_alert_sync, args=(image_local_path, message_text), daemon=True)
    thread.start()
    return True
