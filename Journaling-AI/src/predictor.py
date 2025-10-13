import numpy as np
import torch
from torch.nn.functional import softmax
from .emotion_keywords import fear_keywords, sad_keywords, joy_keywords, anger_keywords

def chunk_text(tokenizer, text, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks


def predict_hybrid_emotion(text, tokenizer_indo, model_indo, tokenizer_xlmr, model_xlmr, labels, id2label):
    model_indo.eval()
    model_xlmr.eval()

    chunks = chunk_text(tokenizer_xlmr, text)
    all_probs = []

    for chunk in chunks:
        inputs_indo = tokenizer_indo(chunk, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            probs_indo = softmax(model_indo(**inputs_indo).logits, dim=-1).numpy()[0]

        inputs_xlmr = tokenizer_xlmr(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            probs_xlmr = softmax(model_xlmr(**inputs_xlmr).logits, dim=-1).numpy()[0]

        avg_probs = (probs_indo + probs_xlmr) / 2
        all_probs.append(avg_probs)

    mean_probs = np.mean(all_probs, axis=0)
    dominant = labels[np.argmax(mean_probs)]

    return {
        "dominant_emotion": dominant,
        "confidence": float(np.max(mean_probs)),
        "emotion_distribution": {labels[i]: float(mean_probs[i]) for i in range(len(labels))},
        "chunk_count": len(chunks)
    }
