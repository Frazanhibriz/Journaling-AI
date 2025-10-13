import streamlit as st
from streamlit_lottie import st_lottie
import plotly.express as px
import requests
from src.predictor import predict_hybrid_emotion
from src.utils import load_model, highlight_emotion_words

st.set_page_config(
    page_title="AI Journaling Emotion Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #0f172a, #1e293b);
        color: #f1f5f9;
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 40%, #1e293b 100%);
        border-right: 2px solid rgba(56,189,248,0.4);
        box-shadow: 5px 0 15px rgba(56,189,248,0.15);
        backdrop-filter: blur(12px);
        color: #f8fafc;
    }
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 1rem;
    }
    .emotion-box {
        background: rgba(30,41,59,0.85);
        border: 1px solid rgba(56,189,248,0.4);
        border-radius: 1rem;
        padding: 1.2rem;
        text-align: center;
        color: #f1f5f9;
        box-shadow: 0 0 20px rgba(56,189,248,0.15);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .emotion-highlight {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
        text-shadow: 0 0 8px rgba(56,189,248,0.5);
    }
    .stButton>button {
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.3rem;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

def load_lottie(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_ai = load_lottie("https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json")

st.sidebar.header("🧭 Tentang Aplikasi")
st.sidebar.markdown("""
AI ini menganalisis **emosi dominan** dari tulisan harianmu menggunakan:
- 🧠 *Hybrid IndoBERTweet + XLM-R*
- 💬 *Analisis kata emosional*
- 📊 *Visualisasi distribusi emosi interaktif*
""")
st.sidebar.divider()
st.sidebar.caption("💡 Tulis jurnal dengan detail agar hasil analisis makin akurat.")

tokenizer_indo, model_indo, tokenizer_xlmr, model_xlmr, labels, id2label = load_model()

col1, col2 = st.columns([1.5, 1])
with col1:
    st.title("🧠 AI Journaling Emotion Analyzer")
    st.markdown("Tuliskan jurnal harianmu di bawah ini dan biarkan AI memahami suasana hatimu 💭")

with col2:
    if lottie_ai:
        st_lottie(lottie_ai, height=200, key="ai_emo")

user_input = st.text_area(
    "📝 Tulis jurnalmu di sini:",
    height=180,
    placeholder="Contoh: Malam ini aku merasa gelisah dan tidak tenang. Suara berderit di lorong membuatku menahan napas...",
)

if st.button("🔍 Analisis Emosi"):
    if user_input.strip():
        with st.spinner("Sedang menganalisis emosi... 🔄"):
            result = predict_hybrid_emotion(
                user_input, tokenizer_indo, model_indo, tokenizer_xlmr, model_xlmr, labels, id2label
            )

        st.markdown(f"""
        <div class="emotion-box">
            ✨ Emosi Dominan: <span class="emotion-highlight">{result['dominant_emotion'].upper()}</span><br>
            Confidence: {result['confidence']:.2f}  
            <br>Jumlah potongan teks: {result['chunk_count']}
        </div>
        """, unsafe_allow_html=True)

        dist = result["emotion_distribution"]
        df_plot = (
            px.data.tips()
        )  # placeholder replaced below

        import pandas as pd
        df_plot = pd.DataFrame({
            "Emotion": list(dist.keys()),
            "Confidence": list(dist.values())
        })
        fig = px.bar(
            df_plot,
            x="Emotion",
            y="Confidence",
            color="Emotion",
            color_discrete_sequence=["#3b82f6", "#06b6d4", "#38bdf8", "#a855f7"],
            title="📊 Distribusi Emosi (Interaktif)",
            text_auto=".2f"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
            title_x=0.25,
            hoverlabel=dict(bgcolor="#1e293b", font_color="white"),
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

        emo = result["dominant_emotion"]
        if emo == "sad":
            st.info("💭 Kamu tampak sedih. Tidak apa-apa, beri waktu untuk dirimu sendiri.")
        elif emo == "joy":
            st.success("🌞 Kamu terlihat bahagia! Pertahankan energi positifmu hari ini 💪")
        elif emo == "anger":
            st.warning("🔥 Sepertinya ada kemarahan. Coba tenangkan diri sejenak.")
        elif emo == "fear":
            st.info("😟 Ada rasa takut atau khawatir. Fokuslah pada hal-hal yang bisa kamu kendalikan.")
    else:
        st.error("Tolong isi teks jurnal terlebih dahulu sebelum dianalisis.")

st.divider()
st.caption("🌙 Dibangun dengan ❤️ menggunakan IndoBERTweet + XLM-R + Streamlit + Plotly")
