# 1. Chọn môi trường gốc (Base Image) - Dùng bản slim cho nhẹ máy
FROM python:3.11-slim

# 2. Đặt thư mục làm việc mặc định bên trong container
WORKDIR /app

# 3. Copy file danh sách thư viện và cài đặt (tối ưu layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ mã nguồn API của bạn vào container
COPY . .

# 5. Mở cổng 5000 (cổng mặc định của Flask)
EXPOSE 5000

# 6. Lệnh để khởi chạy API khi container được bật lên
CMD ["python", "app.py"]