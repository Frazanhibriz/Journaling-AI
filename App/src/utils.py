import re
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .emotion_keywords import fear_keywords, sad_keywords, joy_keywords, anger_keywords



@st.cache_resource
def load_model():
    indo_path = "Frazanhibriz/indobertweet_journal_ensemble_v1"
    xlmr_path = "Frazanhibriz/xlmr_journal_local_v1"

    tokenizer_indo = AutoTokenizer.from_pretrained(indo_path)
    model_indo = AutoModelForSequenceClassification.from_pretrained(indo_path).to("cpu")

    tokenizer_xlmr = AutoTokenizer.from_pretrained(xlmr_path)
    model_xlmr = AutoModelForSequenceClassification.from_pretrained(xlmr_path).to("cpu")

    labels = ["anger", "fear", "joy", "sad"]
    id2label = {i: label for i, label in enumerate(labels)}

    return tokenizer_indo, model_indo, tokenizer_xlmr, model_xlmr, labels, id2label



def plot_emotion_distribution(result):
    labels = list(result["emotion_distribution"].keys())
    probs = list(result["emotion_distribution"].values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probs,
        y=labels,
        orientation="h",
        text=[f"{p:.2f}" for p in probs],
        textposition="outside",
        marker_color=["#EF476F", "#FFD166", "#06D6A0", "#118AB2"]
    ))

    fig.update_layout(
        title=f"Distribusi Emosi ({result['dominant_emotion'].upper()})",
        xaxis_title="Probabilitas",
        yaxis_title="Emosi",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=30, t=60, b=50),
        font=dict(size=13)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_emotion_timeline(result):
    if "chunk_details" not in result:
        return

    chunks = [f"Chunk {i+1}" for i in range(result["chunk_count"])]
    data = []

    for i, chunk_info in enumerate(result["chunk_details"]):
        for emotion, prob in chunk_info["probabilities"].items():
            data.append({"Chunk": chunks[i], "Emotion": emotion, "Probability": prob})

    fig = px.line(
        data,
        x="Chunk",
        y="Probability",
        color="Emotion",
        markers=True,
        title="Perubahan Emosi per Chunk (Timeline)"
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        font=dict(size=13),
        margin=dict(l=40, r=40, t=60, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)



def highlight_emotion_words(text: str) -> str:
    def colorize(text, word_list, color):
        for w in word_list:
            pattern = re.compile(rf"\b({re.escape(w)})\b", re.IGNORECASE)
            text = re.sub(pattern, f"<span style='color:{color}; font-weight:bold'>{w}</span>", text)
        return text

    text = colorize(text, fear_keywords, "#FFD166")   # kuning
    text = colorize(text, sad_keywords, "#118AB2")    # biru
    text = colorize(text, anger_keywords, "#EF476F")  # merah
    text = colorize(text, joy_keywords, "#06D6A0")    # hijau
    return text



def reinforce_emotion_by_keywords(result, text):
    text_lower = text.lower()
    probs = result["emotion_distribution"]

    if any(word in text_lower for word in ["takut", "cemas", "khawatir", "gelisah"]):
        probs["fear"] += 0.05
    if any(word in text_lower for word in ["sedih", "kecewa", "murung", "putus asa"]):
        probs["sad"] += 0.05
    if any(word in text_lower for word in ["senang", "bahagia", "ceria", "gembira"]):
        probs["joy"] += 0.05
    if any(word in text_lower for word in ["marah", "kesal", "benci", "muak"]):
        probs["anger"] += 0.05

    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    result["emotion_distribution"] = probs
    result["dominant_emotion"] = max(probs, key=probs.get)
    result["confidence"] = probs[result["dominant_emotion"]]
    return result
