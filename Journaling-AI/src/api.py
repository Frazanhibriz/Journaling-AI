from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .predictor import predict_hybrid_emotion
from .emotion_keywords import fear_keywords, sad_keywords, joy_keywords, anger_keywords
from .utils import reinforce_emotion_by_keywords



app = FastAPI(
    title="AI Journaling Emotion API (Hybrid)",
    description=(
        "API untuk memprediksi emosi dari teks journaling "
        "menggunakan model Hybrid IndoBERTweet + XLM-R."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

indo_path = os.path.join(BASE_DIR, "Model", "indobertweet_journal_ensemble_v1")
xlmr_path = os.path.join(BASE_DIR, "Model", "xlmr_journal_local_v1")

tokenizer_indo = AutoTokenizer.from_pretrained(indo_path)
model_indo = AutoModelForSequenceClassification.from_pretrained(indo_path)
tokenizer_xlmr = AutoTokenizer.from_pretrained(xlmr_path)
model_xlmr = AutoModelForSequenceClassification.from_pretrained(xlmr_path)

model_indo.eval()
model_xlmr.eval()

labels = ["anger", "fear", "joy", "sad"]
id2label = {i: label for i, label in enumerate(labels)}


class JournalInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Welcome to AI Journaling Hybrid API 🚀"}

@app.post("/predict")
def predict_journal(input: JournalInput):
    text = input.text.strip()
    if not text:
        return {"error": "Teks tidak boleh kosong."}


    result = predict_hybrid_emotion(
        text,
        tokenizer_indo, model_indo,
        tokenizer_xlmr, model_xlmr,
        labels, id2label
    )


    result = reinforce_emotion_by_keywords(result, text)

    return {
        "dominant_emotion": result["dominant_emotion"],
        "confidence": result["confidence"],
        "emotion_distribution": result["emotion_distribution"],
        "chunk_count": result["chunk_count"]
    }

# ==========================================
# ✅ Run command example:
# uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
# ==========================================
