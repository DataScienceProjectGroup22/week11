# =============================================================================
# app.py
# =============================================================================
# Run:  streamlit run app.py
# Model: Logistic Regression (F1=0.9645, ROC-AUC=0.9943)
# =============================================================================

import os
import re
import time
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AI Text Detector",
    page_icon   = "🔍",
    layout      = "centered",
    initial_sidebar_state = "collapsed"
)

# ── File paths ────────────────────────────────────────────────────────────────
DATA_DIR      = "/Users/yashaswini11/Desktop/Team_project/project"
LR_MODEL_PATH = os.path.join(DATA_DIR, "best_classical_model.pkl")
TFIDF_PATH    = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")

# ── CSS ───────────────────────────────────────────────────────────────────────
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

    /* Result cards */
    .card-human {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1.2rem 0;
    }
    .card-ai {
        background: #fff1f2;
        border: 2px solid #dc2626;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1.2rem 0;
    }
    .card-label {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #4b5563;
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

# ── Load model (cached — loads once, reused for every prediction) ─────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load Logistic Regression model and TF-IDF vectorizer from disk."""
    model = joblib.load(LR_MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf

# ── Text preprocessing (matches file 03 exactly) ─────────────────────────────
def preprocess(text: str) -> str:
    """Clean and lemmatise text for TF-IDF input."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet',   quiet=True)
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

# ── Stylometric features ──────────────────────────────────────────────────────
def get_stylometrics(text: str) -> dict:
    """Compute document-level stylometric features from raw text."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    avg_sent  = (sum(len(s.split()) for s in sentences) / len(sentences)
                 if sentences else 0)
    punct     = sum(1 for c in text if c in string.punctuation)
    tokens    = text.split()
    ttr       = (len(set(t.lower() for t in tokens)) / len(tokens)
                 if tokens else 0)
    return {
        "Words"                : len(tokens),
        "Sentences"            : len(sentences),
        "Avg sentence length"  : round(avg_sent, 1),
        "Punctuation density"  : round(punct / len(text) if text else 0, 4),
        "Vocabulary diversity" : round(ttr, 4),
    }

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(text: str, model, tfidf):
    """
    Run TF-IDF + LR classification on input text.

    Returns:
        label      : 0 (Human) or 1 (AI)
        prob_ai    : float — probability of AI-generated
        top_feats  : list of (word, score) tuples — top 5 contributing words
    """
    cleaned       = preprocess(text)
    vec           = tfidf.transform([cleaned])
    prob          = model.predict_proba(vec)[0]
    label         = int(np.argmax(prob))
    prob_ai       = float(prob[1])

    # Top 5 TF-IDF features weighted by LR coefficient
    feature_names = tfidf.get_feature_names_out()
    coef          = model.coef_[0]
    indices       = vec.nonzero()[1]
    if len(indices) > 0:
        scored    = [(feature_names[i], float(coef[i] * vec[0, i]))
                     for i in indices]
        top_feats = sorted(scored, key=lambda x: abs(x[1]), reverse=True)[:5]
    else:
        top_feats = []

    return label, prob_ai, top_feats

# ── UI — Title ────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🔍 AI Text Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Find out if a piece of text was written by a human or generated by AI'
    '</div>',
    unsafe_allow_html=True
)

# ── Text input ────────────────────────────────────────────────────────────────
text = st.text_area(
    label       = "Paste your text here",
    height      = 220,
    placeholder = (
        "Paste any text — an essay, email, article, social media post...\n\n"
        "Minimum 10 words required."
    ),
    label_visibility = "collapsed"
)

word_count = len(text.split()) if text.strip() else 0
if text.strip():
    st.caption(f"{word_count} words")

# ── Analyse button ────────────────────────────────────────────────────────────
analyse = st.button("Analyse Text", type="primary", use_container_width=True)

# ── Validation + Prediction ───────────────────────────────────────────────────
if analyse:
    if not text.strip():
        st.warning("⚠️  Please paste some text first.")

    elif word_count < 10:
        st.warning(
            f"⚠️  Text too short ({word_count} words). "
            "Please provide at least 10 words for a reliable result."
        )

    else:
        with st.spinner("Analysing..."):
            try:
                model, tfidf = load_model()
                t0           = time.time()
                label, prob_ai, top_feats = predict(text, model, tfidf)
                elapsed      = time.time() - t0
                prob_human   = 1.0 - prob_ai
                confidence   = max(prob_ai, prob_human) * 100

            except FileNotFoundError as e:
                st.error(
                    f"Model file not found: `{e}`\n\n"
                    f"Make sure `best_classical_model.pkl` and "
                    f"`tfidf_vectorizer.pkl` are in:\n`{DATA_DIR}`"
                )
                st.stop()
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        # ── Result card ───────────────────────────────────────────────────────
        if label == 1:
            st.markdown(
                '<div class="card-ai">'
                '<div class="card-label">🤖 AI-Generated</div>'
                '<div class="card-desc">This text shows strong signs of '
                'being written by an AI model.</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="card-human">'
                '<div class="card-label">✍️ Human-Written</div>'
                '<div class="card-desc">This text shows strong signs of '
                'being written by a human.</div>'
                '</div>',
                unsafe_allow_html=True
            )

        # ── Confidence bar ────────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-title">Confidence</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Human", f"{prob_human*100:.1f}%")
        with c2:
            st.metric("AI-Generated", f"{prob_ai*100:.1f}%")

        st.progress(
            float(prob_ai),
            text=f"AI probability · {prob_ai*100:.1f}%"
        )

        # ── Details expander ──────────────────────────────────────────────────
        with st.expander("📊 See detailed analysis"):

            col_feat, col_style = st.columns(2)

            with col_feat:
                st.markdown("**Top contributing words**")
                if top_feats:
                    feat_df = pd.DataFrame(
                        [(w, "→ AI" if s > 0 else "→ Human", round(abs(s), 4))
                         for w, s in top_feats],
                        columns=["Word / Phrase", "Points to", "Weight"]
                    )
                    st.dataframe(feat_df, use_container_width=True,
                                 hide_index=True)
                else:
                    st.info("No vocabulary match found.")

            with col_style:
                st.markdown("**Writing style stats**")
                style = get_stylometrics(text)
                style_df = pd.DataFrame(
                    style.items(), columns=["Metric", "Value"]
                )
                st.dataframe(style_df, use_container_width=True,
                             hide_index=True)

            st.caption(
                f"Response time: {elapsed:.3f}s  ·  "
                f"Model: Logistic Regression  ·  "
                f"F1: 0.9645  ·  ROC-AUC: 0.9943"
            )

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="info-box">'
            '💡 This tool is designed to assist human judgement, not replace it. '
            'No AI detector is 100% accurate. Use results as one signal '
            'among many, not as definitive proof.'
            '</div>',
            unsafe_allow_html=True
        )

# ── Example texts ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Try an example:**")

ex1, ex2 = st.columns(2)

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

with ex1:
    if st.button("✍️ Human example", use_container_width=True):
        st.session_state['_example'] = EXAMPLE_HUMAN
        st.rerun()

with ex2:
    if st.button("🤖 AI example", use_container_width=True):
        st.session_state['_example'] = EXAMPLE_AI
        st.rerun()

# Handle example injection
if '_example' in st.session_state:
    text = st.session_state.pop('_example')
    st.text_area("", value=text, height=220, key="injected",
                 label_visibility="collapsed")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'MSc Data Science · University of Hertfordshire · Group Final Project'
    '</div>',
    unsafe_allow_html=True
)
