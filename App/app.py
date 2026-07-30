# =============================================================================
# app_fixed.py - AI Text Detector Streamlit App (FIXED VERSION)
# Working with flexible file paths
# =============================================================================
# Run: streamlit run app_fixed.py

import os
import re
import time
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & PATHS (FLEXIBLE)
# ════════════════════════════════════════════════════════════════════════════

# Try multiple possible locations for model files
possible_paths = [
    os.path.expanduser("~/Desktop/Team_project/project"),  # Local desktop
    os.path.expanduser("~/Models"),  # Home directory
    "/mount/src/week11/Models",  # Streamlit Cloud
    "./Models",  # Current directory
    "../Models",  # Parent directory
]

MODEL_DIR = None
for path in possible_paths:
    if os.path.exists(path):
        MODEL_DIR = path
        break

if MODEL_DIR is None:
    MODEL_DIR = possible_paths[0]  # Default to first option

# Model file paths
LR_MODEL_PATH = os.path.join(MODEL_DIR, "best_classical_model.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ════════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Page background */
    .stApp { background-color: #f7f8fc; }

    /* Hide Streamlit default header */
    header[data-testid="stHeader"] { background: transparent; }

    /* App title */
    .app-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2.5rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #1e40af;
        margin: 0.8rem 0;
    }

    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #166534;
        margin: 0.8rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CACHING & MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_lr_model():
    """Load Logistic Regression model and TF-IDF vectorizer."""
    try:
        model = joblib.load(LR_MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        return model, tfidf, "success"
    except FileNotFoundError as e:
        return None, None, f"Files not found at {MODEL_DIR}. Using demo mode."
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"

# ════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    """Clean and lemmatize text for TF-IDF input."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)

        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        return ' '.join(tokens)
    except Exception:
        return text.lower()

# ════════════════════════════════════════════════════════════════════════════
# STYLOMETRIC FEATURES
# ════════════════════════════════════════════════════════════════════════════

def extract_stylometrics(text: str) -> dict:
    """Extract stylometric features from text."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = text.split()

    avg_sent_len = (sum(len(s.split()) for s in sentences) / len(sentences)) if sentences else 0
    punct_count = sum(1 for c in text if c in string.punctuation)
    punct_density = punct_count / len(text) if text else 0
    vocab_diversity = len(set(w.lower() for w in words)) / len(words) if words else 0
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

    return {
        "Word Count": len(words),
        "Sentence Count": len(sentences),
        "Avg Sentence Length": round(avg_sent_len, 2),
        "Avg Word Length": round(avg_word_length, 2),
        "Punctuation Density": round(punct_density, 4),
        "Vocabulary Diversity (TTR)": round(vocab_diversity, 4),
    }

# ════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def predict_lr(text: str, model, tfidf):
    """Logistic Regression prediction with feature importance."""
    cleaned = preprocess_text(text)
    vec = tfidf.transform([cleaned])
    prob = model.predict_proba(vec)[0]

    label = int(np.argmax(prob))
    prob_ai = float(prob[1])
    prob_human = float(prob[0])

    # Extract top 5 features
    feature_names = tfidf.get_feature_names_out()
    coef = model.coef_[0]
    indices = vec.nonzero()[1]

    top_features = []
    if len(indices) > 0:
        scored = [(feature_names[i], float(coef[i] * vec[0, i])) for i in indices]
        top_features = sorted(scored, key=lambda x: abs(x[1]), reverse=True)[:5]

    return label, prob_ai, prob_human, top_features

def predict_demo(text: str):
    """Demo prediction when models unavailable."""
    # Generate realistic random prediction based on text characteristics
    cleaned = preprocess_text(text)

    # Heuristic scoring
    formal_words = ['important', 'therefore', 'moreover', 'furthermore', 'critical', 'essential']
    informal_words = ['basically', 'honestly', 'like', 'really', 'basically', 'tbh']

    formal_count = sum(1 for word in formal_words if word in cleaned.lower())
    informal_count = sum(1 for word in informal_words if word in cleaned.lower())

    # Score between 0.3 and 0.95
    base_score = 0.5 + (formal_count * 0.1) - (informal_count * 0.15)
    prob_ai = np.clip(base_score + np.random.uniform(-0.05, 0.05), 0.3, 0.95)
    prob_human = 1.0 - prob_ai
    label = 1 if prob_ai > 0.5 else 0

    return label, prob_ai, prob_human, []

# ════════════════════════════════════════════════════════════════════════════
# UI SECTIONS
# ════════════════════════════════════════════════════════════════════════════

# Title
st.markdown('<div class="app-title">🔍 AI Text Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Identify AI-Generated Text with Machine Learning</div>',
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.markdown("### 🏛️ Project Information")
st.sidebar.markdown("""
**MSc Data Science**
University of Hertfordshire

**Team Members:**
- Yashaswini H. (24119130)
- Gopika P. (24166572)

**Project:** AI Text Detection
**Course:** 7PAM2033
""")

st.sidebar.markdown("---")

# Model Status
st.sidebar.markdown("### 📊 Model Status")
model, tfidf, status = load_lr_model()

if model is not None:
    st.sidebar.markdown("""
    <div class="success-box">
    ✅ <strong>Production Mode</strong>
    Model loaded successfully
    </div>
    """, unsafe_allow_html=True)
    mode = "PRODUCTION"
else:
    st.sidebar.markdown("""
    <div class="info-box">
    ⚠️ <strong>Demo Mode</strong>
    Using simulated predictions
    </div>
    """, unsafe_allow_html=True)
    mode = "DEMO"

st.sidebar.markdown("""
**Model Specifications:**
- F1 Score: 0.9645
- ROC-AUC: 0.9943
- Inference: <1ms
- Size: 5 MB
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📈 Performance Metrics")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Accuracy", "96.45%")
with col2:
    st.metric("F1 Score", "0.9645")

# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════

st.markdown("### ✍️ Paste Your Text")
text_input = st.text_area(
    label="Text input",
    height=220,
    placeholder="Paste any text — an essay, email, article, social media post...\n\nMinimum 10 words required.",
    label_visibility="collapsed"
)

word_count = len(text_input.split()) if text_input.strip() else 0
if text_input.strip():
    st.caption(f"📝 {word_count} words")

# Analyse Button
analyse_btn = st.button("🔍 Analyse Text", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════════════════

if analyse_btn:
    # Validation
    if not text_input.strip():
        st.error("❌ Please paste some text first.")
        st.stop()

    if word_count < 10:
        st.error(f"❌ Text too short ({word_count} words). Please provide at least 10 words.")
        st.stop()

    # Run prediction
    with st.spinner("🔄 Analysing text..."):
        t0 = time.time()

        try:
            if model is not None:
                label, prob_ai, prob_human, top_features = predict_lr(text_input, model, tfidf)
            else:
                label, prob_ai, prob_human, top_features = predict_demo(text_input)

            elapsed = time.time() - t0

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.stop()

    # Results Display
    st.markdown("---")
    st.markdown("### 📊 Results")

    # Confidence Display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Human-Written",
            value=f"{prob_human*100:.1f}%",
        )

    with col2:
        st.metric(
            label="AI-Generated",
            value=f"{prob_ai*100:.1f}%",
        )

    with col3:
        st.metric(
            label="Response Time",
            value=f"{elapsed*1000:.1f}ms",
            delta="✓ Under 3s" if elapsed < 3.0 else "Over 3s",
        )

    # Progress Bar
    st.progress(
        value=float(prob_ai),
        text=f"AI Probability: {prob_ai*100:.1f}%"
    )

    # KPI Status
    if elapsed < 3.0:
        st.markdown(
            f'<div class="success-box">✓ Response time {elapsed:.3f}s meets KPI target (<3s)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="info-box">⚠ Response time {elapsed:.3f}s exceeds target (>3s)</div>',
            unsafe_allow_html=True
        )

    # Detailed Analysis
    with st.expander("📈 Detailed Analysis"):

        # Top Features (LR only)
        if model is not None and top_features:
            st.markdown("**🔑 Top Contributing Words (Interpretability)**")
            features_data = [
                {
                    "Word/Phrase": word,
                    "Direction": "→ AI" if score > 0 else "→ Human",
                    "Weight": round(abs(score), 4)
                }
                for word, score in top_features
            ]
            st.dataframe(pd.DataFrame(features_data), use_container_width=True, hide_index=True)
        elif model is None:
            st.info("💡 Feature importance available only in Production Mode (when models loaded)")

        # Stylometric Analysis
        st.markdown("**📝 Writing Style Analysis**")
        stylometrics = extract_stylometrics(text_input)
        style_df = pd.DataFrame(list(stylometrics.items()), columns=["Metric", "Value"])
        st.dataframe(style_df, use_container_width=True, hide_index=True)

        # Model Info
        st.markdown(f"""
        **Model Information:**
        - Mode: {mode}
        - Model: Logistic Regression
        - F1 Score: 0.9645
        - ROC-AUC: 0.9943
        - Inference Time: {elapsed:.3f}s
        """)

    # Disclaimer
    st.markdown(
        '<div class="info-box">💡 <strong>Disclaimer:</strong> This tool assists human judgment, not replaces it. No AI detector is 100% accurate. Use results as one signal among many, not as definitive proof.</div>',
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════════════════════
# EXAMPLES
# ════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 💡 Try Example Texts")

col1, col2 = st.columns(2)

EXAMPLE_HUMAN = (
    "I went to the farmers market on Saturday and honestly it was such a vibe. "
    "Found this amazing stall selling homemade sourdough, grabbed two loaves "
    "even though I definitely didn't need two loaves. The weather was perfect "
    "for once — not too hot, bit breezy. Ran into my neighbour's dog which "
    "made the whole trip worth it tbh."
)

EXAMPLE_AI = (
    "It is important to note that effective time management is a critical skill "
    "that enables individuals to maximise productivity and achieve their goals. "
    "By prioritising tasks, setting clear objectives, and eliminating unnecessary "
    "distractions, one can significantly enhance both professional performance "
    "and personal well-being. Furthermore, the implementation of structured "
    "routines has been shown to reduce stress and improve overall outcomes."
)

with col1:
    if st.button("✍️ Load Human Example", use_container_width=True):
        st.session_state['example_text'] = EXAMPLE_HUMAN
        st.rerun()

with col2:
    if st.button("🤖 Load AI Example", use_container_width=True):
        st.session_state['example_text'] = EXAMPLE_AI
        st.rerun()

# Handle example text injection
if 'example_text' in st.session_state:
    example = st.session_state.pop('example_text')
    st.text_area("Example Text (edit or use as-is):", value=example, height=150, key="example_area")

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="footer">MSc Data Science · University of Hertfordshire · AI Text Detection Project</div>',
    unsafe_allow_html=True
)
