import json
import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    data = {'ticker': 'GOOGL'}
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert 'predicted_price' in response_data, "Response should contain predicted_price"
