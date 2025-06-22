from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_main():
    """
    Testa o endpoint raiz da API.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Smart Meeting API - Bem-vindo!"}


def test_health_check():
    """
    Testa o endpoint de health check.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"} 