"""Private R2 configuration, independent of camera/detection configuration."""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

APP_DIR = Path(__file__).resolve().parent
MAX_URL_TTL = 4 * 24 * 60 * 60


class StorageError(Exception):
    """Safe, actionable message; never includes credentials or signed URLs."""


@dataclass(frozen=True)
class R2Settings:
    endpoint_url: str
    bucket_name: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    key_prefix: str = "alerts"
    url_ttl_seconds: int = MAX_URL_TTL
    public_base_url: str = ""

    @classmethod
    def load(cls, config_path=None, environ=None):
        env = os.environ if environ is None else environ
        path = Path(config_path or env.get("R2_CONFIG_FILE") or APP_DIR / "r2_config.json")
        if not path.is_absolute():
            path = APP_DIR / path
        data = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            pass  # A deployment may use environment variables only.
        except (OSError, ValueError):
            raise StorageError("Không đọc được r2_config.json; kiểm tra quyền và cú pháp JSON.") from None
        if not isinstance(data, dict):
            raise StorageError("Cấu hình R2 phải là một JSON object.")

        names = {
            "endpoint_url": "R2_ENDPOINT_URL",
            "bucket_name": "R2_BUCKET_NAME",
            "access_key_id": "R2_ACCESS_KEY_ID",
            "secret_access_key": "R2_SECRET_ACCESS_KEY",
            "key_prefix": "R2_KEY_PREFIX",
            "url_ttl_seconds": "R2_URL_TTL_SECONDS",
            "public_base_url": "R2_PUBLIC_BASE_URL",
        }
        if set(data) - set(names):
            raise StorageError("Cấu hình R2 có trường không hỗ trợ; đối chiếu file mẫu.")
        values = {key: env.get(var, data.get(key)) for key, var in names.items()}
        for key in ("endpoint_url", "bucket_name", "access_key_id", "secret_access_key"):
            value = values[key]
            if not isinstance(value, str) or not value.strip() or "<" in value:
                raise StorageError(f"Thiếu hoặc chưa điền cấu hình R2: {key}.")
            values[key] = value.strip()

        endpoint = values["endpoint_url"].rstrip("/")
        try:
            parsed = urlsplit(endpoint)
            valid = (
                parsed.scheme == "https"
                and re.fullmatch(
                    r"[a-fA-F0-9]{32}(?:\.[a-z0-9-]+)?\.r2\.cloudflarestorage\.com", parsed.netloc
                )
                and parsed.path == ""
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise StorageError("Endpoint phải là HTTPS S3 endpoint R2 do Cloudflare cung cấp.")
        bucket = values["bucket_name"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
            raise StorageError("Tên bucket R2 không hợp lệ.")
        prefix = values["key_prefix"] if values["key_prefix"] is not None else "alerts"
        if (
            not isinstance(prefix, str)
            or not re.fullmatch(r"[a-zA-Z0-9/_-]{1,100}", prefix)
            or "//" in prefix
        ):
            raise StorageError("key_prefix chỉ nhận chữ, số, dấu /, _ và -.")
        prefix = prefix.strip("/")
        if not prefix:
            raise StorageError("key_prefix không được rỗng.")
        ttl = values["url_ttl_seconds"]
        if ttl is None:
            ttl = MAX_URL_TTL
        if isinstance(ttl, bool) or not re.fullmatch(r"[0-9]+", str(ttl)) or not 1 <= int(ttl) <= MAX_URL_TTL:
            raise StorageError("url_ttl_seconds phải từ 1 đến 345600 giây (4 ngày).")
        public_base = values["public_base_url"]
        if public_base is None or (isinstance(public_base, str) and not public_base.strip()):
            public_base = ""
        elif not isinstance(public_base, str) or "<" in public_base:
            raise StorageError("public_base_url chưa hợp lệ; cần URL HTTPS (ví dụ https://pub-xxx.r2.dev).")
        else:
            public_base = public_base.strip().rstrip("/")
            parsed_pub = urlsplit(public_base)
            if (
                parsed_pub.scheme != "https"
                or not parsed_pub.netloc
                or "@" in parsed_pub.netloc
                or parsed_pub.query
                or parsed_pub.fragment
            ):
                raise StorageError("public_base_url phải là URL HTTPS hợp lệ, không chứa query/fragment.")
        return cls(
            endpoint,
            bucket,
            values["access_key_id"],
            values["secret_access_key"],
            prefix,
            int(ttl),
            public_base,
        )
