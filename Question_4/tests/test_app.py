import pytest
from app import app  # Đảm bảo import ứng dụng Flask từ đúng tệp

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test the index page"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Upload Image for Prediction' in response.data

def test_predict_image(client):
    """Test image prediction"""
    with open('tests/sample_image.png', 'rb') as img:
        response = client.post('/predict', content_type='multipart/form-data', data={
            'image': img
        })
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert isinstance(data['prediction'], int)
