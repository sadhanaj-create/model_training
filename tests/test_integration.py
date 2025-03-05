import json
import pytest
import subprocess
from app import app

@pytest.fixture
def prepare_data():
    subprocess.run(['python3', 'data_preprocessing.py'])
    subprocess.run(['python3', 'model.py'])

@pytest.fixture
def client(prepare_data):
    with app.test_client() as client:
        yield client

def test_integration(client):
    data = {'ticker': 'GOOGL'}
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert 'predicted_price' in response_data
