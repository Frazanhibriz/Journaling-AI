# 🧠 AI-Powered Journaling Emotion Analyzer

> **End-to-End NLP System** untuk menganalisis emosi dari teks journaling Bahasa Indonesia menggunakan pendekatan **Hybrid Ensemble (IndoBERTweet + XLM-RoBERTa)**.

---

## 🌟 Overview
Proyek ini dikembangkan untuk membantu individu memahami kondisi emosional mereka melalui journaling. AI ini mampu mendeteksi 4 emosi dasar (**Anger, Fear, Joy, Sad**) dengan akurasi tinggi, bahkan pada teks yang panjang dan menggunakan bahasa sehari-hari (informal).

### Key Features:
- **Hybrid Architecture:** Menggabungkan kekuatan **IndoBERTweet** (slang-heavy) dan **XLM-RoBERTa** (context-heavy).
- **Long-Text Support:** Implementasi *Sliding Window Chunking* untuk memproses teks jurnal yang melebihi 512 token.
- **Rule-Based Reinforcement:** Integrasi logika berbasis leksikon untuk menstabilkan prediksi model deep learning.
- **Interactive Dashboard:** Visualisasi distribusi emosi yang interaktif menggunakan Streamlit dan Plotly.

---

## 🏗️ Architecture & Methodology

### 1. Hybrid Ensemble Strategy
Mengapa menggunakan dua model?
*   **IndoBERTweet:** Sangat optimal untuk menangani teks informal, slang, dan singkatan khas Indonesia.
*   **XLM-RoBERTa:** Memberikan pemahaman konteks semantik yang lebih luas dan stabil pada struktur kalimat kompleks.
*   **Fusion Logic:** Sistem melakukan *Averaging Softmax Probabilities* dari kedua model untuk mendapatkan hasil akhir yang lebih robust dibandingkan model tunggal.

### 2. Handling Long-Form Journaling (Chunking)
Model Transformer memiliki batasan 512 token. Untuk menangani jurnal yang panjang, sistem ini secara otomatis:
1.  Membagi teks menjadi beberapa **chunks** (potongan).
2.  Melakukan inferensi pada setiap potongan secara independen.
3.  Melakukan agregat hasil menggunakan **Mean Probability** untuk menentukan emosi dominan secara keseluruhan.

### 3. Heuristic Reinforcement
Untuk meningkatkan *confidence*, sistem dilengkapi dengan *Emotion Keywords Reinforcement* yang memberikan bobot tambahan jika ditemukan kata-kata emosional eksplisit (seperti "kecewa", "bahagia", "takut") dalam teks.

---

## 📊 Performance (Internal Benchmark)
Sistem ini telah diuji menggunakan dataset journaling yang disingkronkan secara lokal dengan hasil sebagai berikut:

| Metric | Score |
|:---|:---|
| **Overall Accuracy** | **91%** |
| **Avg. F1-Score** | **0.91** |

**F1-Score per Category:**
- 🔴 **Anger:** 0.92
- 🟡 **Fear:** 0.92
- 🟢 **Joy:** 0.93
- 🔵 **Sad:** 0.89

---

## 🛠️ Technical Stack
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Models:** [IndoBERTweet](https://huggingface.co/Frazanhibriz/indobertweet_journal_ensemble_v1) & [XLM-RoBERTa](https://huggingface.co/Frazanhibriz/xlmr_journal_v1)
- **Deployment:** FastAPI (Backend) & Streamlit (Frontend)
- **Visualization:** Plotly, Pandas

---

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/Frazanhibriz/Journaling-AI.git
cd Journaling-AI

# Install dependencies
pip install -r App/requirements.txt
