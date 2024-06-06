from fastapi.testclient import TestClient
from .chat_megumi import app

client = TestClient(app)

def test_create_token():
    response = client.post("/token", headers={"Content-Type": "application/x-www-form-urlencoded"}, data="username=root&password=root")
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "token_type" in data
    assert "expires_at" in data
    # test failed case
    response = client.post("/token", headers={"Content-Type": "application/x-www-form-urlencoded"}, data="username=root&password=root123")
    assert response.status_code == 400

def test_chat():
    response = client.post("/token", headers={"Content-Type": "application/x-www-form-urlencoded"}, data="username=root&password=root")
    token = response.json()["access_token"]
    response = client.post("/api/chat", headers={"Authorization": f"Bearer {token}"}, json={"content": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "translation" in data
    # test failed case
    response = client.post("/api/chat", json={"content": "Hello"})
    assert response.status_code == 401