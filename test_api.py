import requests
import pytest

BASE_URL = "http://localhost:8000"  # Для локального тестирования
# BASE_URL = "https://your-api-url.com"  # Для тестирования на сервере

def test_textblob_analysis():
    """Тест анализа через TextBlob."""
    response = requests.post(
        f"{BASE_URL}/analyze/textblob",
        json={"text": "I love Python!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] in ["Positive", "Neutral", "Negative"]

def test_huggingface_analysis():
    """Тест анализа через Hugging Face."""
    response = requests.post(
        f"{BASE_URL}/analyze/huggingface",
        json={"text": "This is terrible!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] in ["POSITIVE", "NEGATIVE"]

def test_invalid_input():
    """Тест на некорректный ввод."""
    response = requests.post(
        f"{BASE_URL}/analyze/textblob",
        json={"wrong_key": "test"}
    )
    assert response.status_code == 422  # Ошибка валидации FastAPI