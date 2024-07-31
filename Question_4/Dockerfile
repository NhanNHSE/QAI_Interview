# Sử dụng hình ảnh python 3.9
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements.txt nếu có (nếu không có thì có thể bỏ qua phần này)
COPY requirements.txt .

# Cài đặt các gói phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

# Copy tất cả các tệp dự án vào trong container
COPY . .

# Expose cổng mà ứng dụng sẽ chạy
EXPOSE 8000

# Chạy ứng dụng FastAPI bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
