# Cảnh Báo Xe Ra/Vào Xưởng (Cloudflare R2 + Zalo Bot)

Hệ thống AI nhận diện xe ra/vào xưởng qua camera RTSP, lưu trữ ảnh cảnh báo trên Cloudflare R2 (tự động xóa sau 4 ngày) và gửi tin nhắn cảnh báo trực tiếp tới Zalo Bot.

## Cấu trúc thư mục

```text
├── main.py                     # Chương trình nhận diện chính (YOLOv8 + BoT-SORT)
├── tracker_botsort.yaml        # Cấu hình thuật toán theo dõi xe
├── yolov8s.pt                  # Model YOLOv8s
├── config.json                 # Cấu hình camera RTSP & Zalo Bots
├── r2_config.json              # Cấu hình lưu trữ ảnh Cloudflare R2
├── zalo_utils_refactored.py    # Gửi thông báo & ảnh tới Zalo
├── image_storage.py            # Upload ảnh lên R2
├── r2_settings.py              # Đọc & kiểm tra cấu hình R2
└── requirements.txt            # Thư viện phụ thuộc bổ sung (boto3)
```

## Hướng dẫn triển khai trên VPS (Ubuntu/Debian)

### 1. Clone mã nguồn về VPS
```bash
git clone https://github.com/Vanhaoqb146/CanhBaoXeRaVaoXuong.git
cd CanhBaoXeRaVaoXuong
```

### 2. Cài đặt thư viện bổ sung
Đảm bảo đang ở môi trường Python 3.11 của hệ thống, cài đặt SDK R2:
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy hệ thống
```bash
python -u main.py
```

### 4. Cập nhật mã nguồn khi có bản mới
Mỗi khi có cập nhật, trên VPS chỉ cần chạy:
```bash
git pull
# Sau đó khởi động lại tiến trình main.py
```
