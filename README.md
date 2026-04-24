# 🧠 AI-Powered Journaling Emotion Analyzer

> An end-to-end NLP system designed to analyze emotional states from Indonesian journaling text using a hybrid transformer-based approach.

---

## 🌟 Overview

This project aims to help individuals better understand their emotional patterns through daily journaling. By leveraging natural language processing, the system classifies text into four primary emotions: **Anger, Fear, Joy, and Sadness**.

The system is specifically designed to handle **informal Indonesian language**, including slang, mixed expressions, and long-form writing, which are commonly found in personal journals.

---

## 🚀 Key Features

- **Hybrid Model Architecture**  
  Combines multiple transformer models to improve prediction stability and accuracy.

- **Long Text Support**  
  Handles journaling entries beyond token limits using a chunk-based processing approach.

- **Heuristic Enhancement**  
  Incorporates keyword-based reinforcement to refine predictions.

- **Interactive Dashboard**  
  Visualizes emotion distribution through an intuitive interface.

---

## 🏗️ Methodology

### 1. Hybrid Ensemble Approach

This project utilizes two pretrained models with complementary strengths:

- **IndoBERTweet**  
  Optimized for informal Indonesian text, including slang and conversational writing.

- **XLM-RoBERTa**  
  Provides stronger contextual understanding for more structured and complex sentences.

The final prediction is obtained by averaging the softmax probabilities from both models, resulting in more robust outputs compared to a single-model approach.

---

### 2. Handling Long Text Inputs

Transformer models are limited to 512 tokens. To address this, the system implements a **sliding window strategy**:

1. Split input text into smaller segments  
2. Perform inference on each segment  
3. Aggregate predictions using mean probability  

This approach ensures consistent performance even on lengthy journal entries.

---

### 3. Rule-Based Reinforcement

To complement model predictions, a lightweight rule-based mechanism is applied.

If explicit emotional keywords (e.g., *“bahagia”*, *“takut”*, *“kecewa”*) are detected, the system slightly adjusts prediction confidence. This helps reduce misclassification in certain edge cases.

---

## 📊 Performance

The model was evaluated on a locally curated Indonesian journaling dataset.

| Metric | Score |
|------|------|
| Accuracy | 91% |
| Average F1-Score | 0.91 |

**F1-Score by Class:**
- Anger: 0.92  
- Fear: 0.92  
- Joy: 0.93  
- Sadness: 0.89  

---

## 🛠️ Tech Stack

- **Frameworks:** PyTorch, Hugging Face Transformers  
- **Models:** IndoBERTweet, XLM-RoBERTa  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Visualization:** Plotly, Pandas  

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Frazanhibriz/Journaling-AI.git

# Navigate to project directory
cd Journaling-AI

# Install dependencies
pip install -r App/requirements.txt
