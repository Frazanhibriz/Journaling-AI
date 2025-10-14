# 🧠 AI Journaling Emotion API

> FastAPI-based emotion analysis backend for journaling texts using IndoBERTweet + XLM-R hybrid model.

---

## 📘 Overview

This API predicts emotional states (anger, fear, joy, sad) from user journaling text in Indonesian.
It combines two fine-tuned transformer models **IndoBERTweet** and **XLM-R** to improve robustness and contextual understanding.

---

## 🚀 Features

- Hybrid ensemble model (IndoBERTweet + XLM-R)
- Emotion classification: `anger`, `fear`, `joy`, `sad`
- Keyword-based emotion reinforcement
- CORS-ready for web integration
- Deployable to Render, Railway, or GCP

---

## 🧩 Model Sources

| Model | Location | Description |
|:------|:----------|:-------------|
| `Frazanhibriz/indobertweet_journal_ensemble_v1` | [Hugging Face Hub](https://huggingface.co/Frazanhibriz/indobertweet_journal_ensemble_v1) | IndoBERTweet fine-tuned for journaling emotion |
| `Frazanhibriz/xlmr_journal_v1` | [Hugging Face Hub](https://huggingface.co/Frazanhibriz/xlmr_journal_v1) | XLM-R fine-tuned for multi-language emotion context |

---

## ⚙️ Setup & Run Locally

```bash
# Clone the repo
git clone https://github.com/Frazanhibriz/Journaling-AI-API.git
cd Journaling-AI-API

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
