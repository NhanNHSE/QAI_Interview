# test/test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app  # Import ứng dụng FastAPI từ file main.py

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Upload and Predict" in response.text

def test_predict():
    # Thay thế với file ảnh thực sự để kiểm tra
    response = client.post("/predict/", files={"file": ("test_image.png", open("test_image.png", "rb"))})
    assert response.status_code == 200
