# AI Text Detection & Rewrite System (REST API)

Hệ thống API backend phục vụ 2 chức năng chính:
1. **AI Detection**: Đánh giá phần trăm văn bản tiếng Việt do Trí tuệ nhân tạo (AI) sinh ra bằng mô hình PhoBERT.
2. **AI Rewrite**: Viết lại văn bản đầu vào để né tránh hệ thống phát hiện AI (Bypass Detector) nhưng vẫn giữ nguyên ý nghĩa gốc bằng mô hình ViT5.

---

## 🚀 Tính năng nổi bật
- **REST API (Flask)**: Dễ dàng tích hợp với mọi nền tảng hoặc Frontend (React, Vue, Mobile App).
- **Tự động tải mô hình (HuggingFace Hub)**: 
  - `nqp426/phobert-ai-detect`: Mô hình phân loại văn bản AI/Human.
  - `nqp426/vit5-ai-rewrite`: Mô hình Text-to-Text sinh văn bản.
- **Sliding Window Algorithm**: Xử lý văn bản dài không giới hạn bằng cách trượt cửa sổ n-câu, giúp khoanh vùng chính xác đoạn văn chứa nội dung do AI tạo ra.

---

## 🗂 Cấu trúc thư mục

```text
nckh_new/
├── backend/
│   ├── app.py                  # API Server chính (chạy Flask)
│   ├── config.py               # Chứa tham số cấu hình (ngưỡng xác suất, window size)
│   ├── inference_engine.py     # Lõi xử lý logic AI Detection & Sliding Window
│   ├── model_loader.py         # Thread-safe tải mô hình PhoBERT
│   ├── rewrite_engine.py       # Lõi xử lý logic sinh câu ViT5
│   ├── rewrite_loader.py       # Thread-safe tải mô hình ViT5
│   └── text_processor.py       # Logic xử lý text, tách câu (dùng Underthesea)
│
├── requirements.txt            
├── .gitignore
└── README.md
```

---

## ⚙️ Cài đặt (Installation)

### 1. Yêu cầu hệ thống
- **Python >= 3.11** 
- Khuyên dùng môi trường có hỗ trợ Card Đồ hoạ (CUDA) để API phản hồi tức thì (Real-time).

### 2. Cài đặt chi tiết

```bash
# Clone hoặc tải dự án về máy
git clone <your-repo-url>
cd nckh_new

# Tạo môi trường ảo (Virtual Env)
python -m venv venv

# Kích hoạt môi trường:
# Trên Windows:
venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

> **Lưu ý**: Lần đầu tiên chạy Server, hệ thống sẽ tải các mô hình từ HuggingFace (khoảng ~1.8GB tổng cộng) và lưu cache vào máy. Từ lần thứ 2 trở đi, API sẽ khởi động cực nhanh (1-3s).

---

## � Khởi chạy Server (Usage)

Khởi động Flask API Server mặc định ở cổng `5000`:

```bash
python backend/app.py
```

---

## 📖 Hướng dẫn sử dụng API

### 1. Phân tích văn bản gốc (Detection)
Kiểm tra xem một văn bản có bị thao túng bởi AI hay không.

- **Endpoint:** `POST /api/analyze`
- **Request Body:**
```json
{
  "text": "Bạn cần một đoạn mã để phân loại văn bản tiếng Việt. Tuy nhiên, việc nhận diện nội dung này không hề dễ dàng..."
}
```
- **Response:**
```json
{
  "global_score": 0.82,
  "sentences": [
    {
      "is_ai": true,
      "score": 0.85,
      "text": "Bạn cần một đoạn mã để phân loại văn bản tiếng Việt."
    },
    {
      "is_ai": false,
      "score": 0.12,
      "text": "Tuy nhiên, việc nhận diện nội dung này không hề dễ dàng..."
    }
  ]
}
```

### 2. Viết lại văn bản (Rewrite)
Đuổi các đoạn văn bản AI "lộ liễu" thành văn phong con người.

- **Endpoint**: `POST /api/rewrite`
- **Request Body**:
```json
{
  "target": "Bạn cần một đoạn mã để phân loại văn bản tiếng Việt.",
  "mode": "rewrite" 
}
```
- **Response**:
```json
{
  "rewritten": "Đoạn code phân loại câu tiếng Việt là thứ bạn đang tìm kiếm."
}
```
