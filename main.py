from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from textblob import TextBlob
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os
from typing import Optional

# --- Настройки FastAPI ---
app = FastAPI(
    title="Sentiment Analysis API",
    description="API для анализа тональности текста (TextBlob + HuggingFace)",
    version="1.0"
)

# --- Модель HuggingFace ---
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PATH = "models"

def load_huggingface_model():
    """Загружает модель Hugging Face и сохраняет её локально."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if not os.listdir(MODEL_PATH):
        print("Загрузка модели Hugging Face... (~1-2 мин)")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)
        print("Модель сохранена в папку 'models'.")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

hf_pipeline = load_huggingface_model()

# --- Pydantic модель для запроса ---
class TextRequest(BaseModel):
    text: str

# --- Эндпоинты API ---
@app.post("/analyze/textblob")
def analyze_textblob(request: TextRequest):
    """Анализ тональности через TextBlob."""
    analysis = TextBlob(request.text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        label = "Positive"
    elif polarity < 0:
        label = "Negative"
    else:
        label = "Neutral"
    return {"text": request.text, "label": label, "score": polarity}

@app.post("/analyze/huggingface")
def analyze_huggingface(request: TextRequest):
    """Анализ тональности через Hugging Face."""
    result = hf_pipeline(request.text)[0]
    return {"text": request.text, "label": result["label"], "score": result["score"]}

@app.get("/")
def read_root():
    message = "Sentiment Analysis API. Используйте /analyze/textblob или /analyze/huggingface"
    return JSONResponse(content={"message": message}, media_type="application/json; charset=utf-8")

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}