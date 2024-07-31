import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to my Flask app!" in response.data

def test_predict_no_file(client):
    response = client.post('/predict', data={})
    json_data = response.get_json()
    assert response.status_code == 200
    assert json_data['error'] == 'No file part'

def test_predict_empty_file(client):
    data = {
        'file': (io.BytesIO(b''), '')
    }
    response = client.post('/predict', data=data)
    json_data = response.get_json()
    assert response.status_code == 200
    assert json_data['error'] == 'No selected file'


