"""Upload one small snapshot to private Cloudflare R2 and sign a GET URL."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from r2_settings import R2Settings, StorageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class UploadedImage:
    key: str
    url: str = field(repr=False)
    size: int
    ttl_seconds: int


def read_image(image_path):
    """Bound memory use; accept JPEG/PNG only and do not expose file contents."""
    try:
        with Path(image_path).open("rb") as source:
            body = source.read(MAX_IMAGE_BYTES + 1)
    except (OSError, TypeError, ValueError):
        raise StorageError("Không đọc được ảnh; kiểm tra đường dẫn và quyền truy cập.") from None
    if not body or len(body) > MAX_IMAGE_BYTES:
        raise StorageError("Ảnh phải có dung lượng từ 1 byte đến 20 MiB.")
    if body.startswith(b"\xff\xd8\xff"):
        return body, "image/jpeg", ".jpg"
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return body, "image/png", ".png"
    raise StorageError("Chỉ hỗ trợ ảnh JPEG hoặc PNG.")


class R2Storage:
    def __init__(self, settings, client=None):
        self.settings = settings
        if client is not None:
            self.client = client
            return
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise StorageError("Thiếu boto3; chạy python -m pip install -r requirements-r2.txt.") from None
        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=settings.endpoint_url,
                aws_access_key_id=settings.access_key_id,
                aws_secret_access_key=settings.secret_access_key,
                region_name="auto",
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    connect_timeout=5,
                    read_timeout=20,
                    retries={"mode": "standard", "total_max_attempts": 3},
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            )
        except Exception:
            raise StorageError("Không khởi tạo được R2 client; kiểm tra cấu hình/SDK.") from None

    @classmethod
    def from_config(cls, config_path=None):
        return cls(R2Settings.load(config_path))

    def signed_get(self, key):
        if self.settings.public_base_url:
            clean_key = str(key).lstrip("/")
            if not clean_key:
                raise StorageError("Key ảnh R2 không hợp lệ.")
            return f"{self.settings.public_base_url}/{clean_key}"
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.settings.bucket_name, "Key": key},
                ExpiresIn=self.settings.url_ttl_seconds,
                HttpMethod="GET",
            )
        except Exception:
            raise StorageError("Không tạo được URL GET có chữ ký cho ảnh R2.") from None

    def upload(self, image_path):
        body, content_type, suffix = read_image(image_path)
        date = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        key = f"{self.settings.key_prefix}/{date}/{uuid4().hex}{suffix}"
        # SDK retries overwrite the same key: one logical object, no duplicate upload keys.
        try:
            self.client.put_object(
                Bucket=self.settings.bucket_name,
                Key=key,
                Body=body,
                ContentType=content_type,
                CacheControl=f"public, max-age={self.settings.url_ttl_seconds}",
            )
        except Exception as error:
            response = getattr(error, "response", {})
            details = response.get("Error", {}) if isinstance(response, dict) else {}
            code = details.get("Code") if isinstance(details, dict) else None
            if not isinstance(code, str):
                code = None
            reasons = {
                "AccessDenied": "R2 từ chối quyền truy cập; kiểm tra bucket/quyền Object Read & Write.",
                "InvalidAccessKeyId": "R2 không nhận Access Key ID; kiểm tra khóa còn hiệu lực.",
                "SignatureDoesNotMatch": "Chữ ký R2 không khớp; kiểm tra secret/endpoint và đồng hồ máy.",
                "NoSuchBucket": "Không tìm thấy bucket R2; kiểm tra bucket và endpoint.",
                "SlowDown": "R2 đang giới hạn request; thử lại sau.",
            }
            raise StorageError(
                reasons.get(code, "Upload R2 thất bại; kiểm tra mạng, khóa và quyền Object Read & Write.")
            ) from None
        return UploadedImage(key, self.signed_get(key), len(body), self.settings.url_ttl_seconds)

    def delete(self, key):
        """Used only for explicitly requested cleanup of a diagnostic upload."""
        try:
            self.client.delete_object(Bucket=self.settings.bucket_name, Key=key)
        except Exception:
            raise StorageError("Không xóa được ảnh kiểm tra trên R2.") from None
