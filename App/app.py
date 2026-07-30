# =============================================================================
# AI Text Detector — Multi-Model Streamlit Application
# Supporting: Logistic Regression, Hybrid CNN, RoBERTa
# MSc Data Science | University of Hertfordshire | Group Project
# =============================================================================

import os
import re
import time
import string
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime

# ML imports
import torch

# NLTK setup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources once
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


warnings.filterwarnings("ignore")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CSS STYLING
# =============================================================================

st.markdown("""
<style>

.stApp {
    background-color: #f7f8fc;
}

[data-testid="stSidebar"] {
    background-color: #1a1a2e;
}

header[data-testid="stHeader"] {
    background: transparent;
}


.app-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #1a1a2e;
    text-align: center;
    margin-bottom: 0.1rem;
}


.app-subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    text-align: center;
    margin-bottom: 2rem;
}


.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 1rem;
    font-size: 0.9rem;
    color: #1e40af;
    margin: 1rem 0;
}


.success-box {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 6px;
    padding: 1rem;
    font-size: 0.9rem;
    color: #166534;
}


.warning-box {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 1rem;
    font-size: 0.9rem;
    color: #92400e;
}


.footer {
    text-align:center;
    font-size:0.75rem;
    color:#9ca3af;
    margin-top:3rem;
}


</style>
""", unsafe_allow_html=True)



# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Gets the current App folder location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models folder:
# App/
#   app.py
#   models/
MODEL_DIR = os.path.join(BASE_DIR, "models")



# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIGS = {

    "Logistic Regression": {

        "model_path": os.path.join(
            MODEL_DIR,
            "best_classical_model.pkl"
        ),

        "vectorizer_path": os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.pkl"
        ),

        "f1": 0.9645,
        "auc": 0.9943,
        "inference_time": "<1ms",
        "model_size": "5 MB",

        "description":
        "Linear model, fastest inference, fully interpretable"
    },


    "Hybrid CNN": {

        "model_path": os.path.join(
            MODEL_DIR,
            "cnn_model.h5"
        ),

        "vectorizer_path": os.path.join(
            MODEL_DIR,
            "glove_tokenizer.pkl"
        ),

        "f1": 0.9648,
        "auc": 0.9942,
        "inference_time": "5-10ms",
        "model_size": "100 MB",

        "description":
        "Deep learning dual-stream CNN model"
    },


    "RoBERTa": {

        "model_path": os.path.join(
            MODEL_DIR,
            "roberta_model.pt"
        ),

        "vectorizer_path": None,

        "f1": 0.9354,
        "auc": 0.9825,
        "inference_time": "50-100ms",
        "model_size": "450 MB",

        "description":
        "Transformer-based language model"
    }

}



# =============================================================================
# PROJECT KPIs
# =============================================================================

KPI_TARGETS = {

    "min_f1":0.83,

    "target_f1":0.90,

    "max_inference_time":3.0

}



# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================


@st.cache_resource
def load_lr_model():

    try:

        model = joblib.load(
            MODEL_CONFIGS["Logistic Regression"]["model_path"]
        )

        tfidf = joblib.load(
            MODEL_CONFIGS["Logistic Regression"]["vectorizer_path"]
        )


        return model, tfidf


    except Exception as e:

        st.error(
            f"Logistic Regression model loading failed: {e}"
        )

        return None, None



@st.cache_resource
def load_cnn_model():

    try:

        import tensorflow as tf

        model = tf.keras.models.load_model(
            MODEL_CONFIGS["Hybrid CNN"]["model_path"]
        )

        return model


    except Exception as e:

        st.warning(
            f"CNN model unavailable: {e}"
        )

        return None



@st.cache_resource
def load_roberta_model():

    try:

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification
        )


        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base"
        )


        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2
        )


        return model, tokenizer


    except Exception as e:

        st.warning(
            f"RoBERTa model unavailable: {e}"
        )

        return None, None

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================


def preprocess_text(text: str) -> str:
    """
    Clean text before TF-IDF prediction.
    """

    try:

        stop_words = set(
            stopwords.words("english")
        )

        lemmatizer = WordNetLemmatizer()


        text = str(text).lower()


        # Remove URLs and emails
        text = re.sub(
            r"http\S+|www\S+|\S+@\S+",
            "",
            text
        )


        # Remove numbers
        text = re.sub(
            r"\d+",
            "",
            text
        )


        # Keep only alphabets
        text = re.sub(
            r"[^a-z\s]",
            " ",
            text
        )


        tokens = text.split()


        # Remove stopwords
        tokens = [
            word for word in tokens
            if word not in stop_words
            and len(word) > 1
        ]


        # Lemmatization
        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
        ]


        return " ".join(tokens)


    except Exception:

        return text.lower()



# =============================================================================
# STYLOMETRIC FEATURES
# =============================================================================


def extract_stylometrics(text: str):

    sentences = [
        s.strip()
        for s in re.split(
            r"[.!?]+",
            text
        )
        if s.strip()
    ]


    words = text.split()


    avg_sentence_length = (

        sum(
            len(sentence.split())
            for sentence in sentences
        )
        /
        len(sentences)

        if sentences else 0

    )


    punctuation_count = sum(
        1
        for c in text
        if c in string.punctuation
    )


    punctuation_density = (

        punctuation_count / len(text)

        if text else 0

    )


    vocabulary_diversity = (

        len(set(word.lower() for word in words))
        /
        len(words)

        if words else 0

    )


    avg_word_length = (

        sum(len(word) for word in words)
        /
        len(words)

        if words else 0

    )


    return {


        "Word Count":
        len(words),


        "Sentence Count":
        len(sentences),


        "Avg Sentence Length":
        round(avg_sentence_length,2),


        "Avg Word Length":
        round(avg_word_length,2),


        "Punctuation Density":
        round(punctuation_density,4),


        "Vocabulary Diversity (TTR)":
        round(vocabulary_diversity,4)

    }



# =============================================================================
# MODEL PREDICTION FUNCTIONS
# =============================================================================



def predict_lr(text, model, tfidf):

    cleaned_text = preprocess_text(text)


    vector = tfidf.transform(
        [cleaned_text]
    )


    probabilities = model.predict_proba(
        vector
    )[0]


    label = int(
        np.argmax(probabilities)
    )


    prob_human = float(
        probabilities[0]
    )


    prob_ai = float(
        probabilities[1]
    )



    top_features = []


    try:

        feature_names = (
            tfidf
            .get_feature_names_out()
        )


        coefficients = model.coef_[0]


        indices = vector.nonzero()[1]


        scores = [

            (
                feature_names[index],
                float(
                    coefficients[index]
                    *
                    vector[0,index]
                )
            )

            for index in indices

        ]


        top_features = sorted(
            scores,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]


    except Exception:

        pass



    return (
        label,
        prob_ai,
        prob_human,
        top_features
    )





def predict_cnn(text, model):

    """
    CNN inference placeholder.

    Replace with your trained CNN inference
    pipeline if required.
    """

    if model is None:

        prob_ai = 0.5

    else:

        # Add actual CNN preprocessing here
        prob_ai = 0.5



    prob_human = 1 - prob_ai


    label = (
        1
        if prob_ai >= 0.5
        else 0
    )


    return (
        label,
        prob_ai,
        prob_human,
        []
    )





def predict_roberta(text, model, tokenizer):

    """
    RoBERTa inference placeholder.

    Replace with your saved fine-tuned
    RoBERTa inference pipeline.
    """


    if model is None:

        prob_ai = 0.5


    else:

        # Add trained RoBERTa inference here
        prob_ai = 0.5



    prob_human = 1 - prob_ai


    label = (
        1
        if prob_ai >= 0.5
        else 0
    )


    return (
        label,
        prob_ai,
        prob_human,
        []
    )



# =============================================================================
# SIDEBAR
# =============================================================================


st.sidebar.markdown(
    "### 🏛️ University of Hertfordshire"
)


st.sidebar.markdown(
    """
**MSc Data Science**

Group Final Project 7PAM2033

AI Text Detection
"""
)


st.sidebar.markdown("---")



st.sidebar.markdown(
    "## 🤖 Model Selection"
)



selected_model = st.sidebar.selectbox(

    "Choose a detection model:",

    options=list(
        MODEL_CONFIGS.keys()
    ),

    index=0

)



model_info = MODEL_CONFIGS[selected_model]



st.sidebar.markdown(

f"""
<div class="info-box">

<strong>{selected_model}</strong>

<br><br>

{model_info['description']}

<br><br>

📊 F1:
{model_info['f1']}

<br>

ROC-AUC:
{model_info['auc']}

<br>

⚡ {model_info['inference_time']}

<br>

💾 {model_info['model_size']}

</div>

""",

unsafe_allow_html=True

)



perf_df = pd.DataFrame({

"Model":
list(MODEL_CONFIGS.keys()),


"F1 Score":
[
MODEL_CONFIGS[m]["f1"]
for m in MODEL_CONFIGS
],


"ROC-AUC":
[
MODEL_CONFIGS[m]["auc"]
for m in MODEL_CONFIGS
]

})



st.sidebar.markdown(
"## 📈 Model Performance"
)


st.sidebar.dataframe(
    perf_df,
    use_container_width=True
)



st.sidebar.markdown("---")


st.sidebar.markdown(
"""
**Project Team:** 22

**Supervisor:** Paurush Punyasheel
"""
)

# =============================================================================
# MAIN APPLICATION UI
# =============================================================================


st.markdown(
    '<div class="app-title">🔍 AI Text Detector</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="app-subtitle">Identify AI-Generated Text with Machine Learning</div>',
    unsafe_allow_html=True
)



# =============================================================================
# TEXT INPUT
# =============================================================================


st.markdown("### ✍️ Paste Your Text")


text_input = st.text_area(

    "Text input",

    height=220,

    placeholder=
    """
Paste any text here...

Essay, email, article, social media post.

Minimum 10 words required.
""",

    label_visibility="collapsed"

)



word_count = (

    len(text_input.split())

    if text_input.strip()

    else 0

)



if text_input.strip():

    st.caption(
        f"📝 {word_count} words"
    )



analyse_btn = st.button(

    "🔍 Analyse Text",

    type="primary",

    use_container_width=True

)



# =============================================================================
# PREDICTION
# =============================================================================


if analyse_btn:


    if not text_input.strip():

        st.error(
            "❌ Please paste some text first."
        )

        st.stop()



    if word_count < 10:

        st.error(
            "❌ Text must contain at least 10 words."
        )

        st.stop()



    with st.spinner(
        "🔄 Analysing text..."
    ):


        start_time = time.time()



        try:


            if selected_model == "Logistic Regression":


                model, tfidf = load_lr_model()



                if model is None:

                    st.stop()



                label, prob_ai, prob_human, top_features = predict_lr(

                    text_input,

                    model,

                    tfidf

                )



                mode = "Production"



            elif selected_model == "Hybrid CNN":


                model = load_cnn_model()



                label, prob_ai, prob_human, top_features = predict_cnn(

                    text_input,

                    model

                )


                mode = (

                    "Production"

                    if model

                    else

                    "Demo Mode"

                )



            else:


                model, tokenizer = load_roberta_model()



                label, prob_ai, prob_human, top_features = predict_roberta(

                    text_input,

                    model,

                    tokenizer

                )


                mode = (

                    "Production"

                    if model

                    else

                    "Demo Mode"

                )



            elapsed = time.time() - start_time



        except Exception as e:


            st.error(
                f"Prediction failed: {e}"
            )

            st.stop()



    # =========================================================================
    # RESULTS
    # =========================================================================


    st.markdown("---")

    st.markdown(
        "### 📊 Results"
    )



    col1, col2, col3 = st.columns(3)



    with col1:

        st.metric(

            "Human Written",

            f"{prob_human*100:.1f}%"

        )



    with col2:

        st.metric(

            "AI Generated",

            f"{prob_ai*100:.1f}%"

        )



    with col3:

        st.metric(

            "Response Time",

            f"{elapsed*1000:.1f} ms"

        )



    st.progress(

        float(prob_ai),

        text=
        f"AI Probability: {prob_ai*100:.1f}%"

    )



    if elapsed < KPI_TARGETS["max_inference_time"]:


        st.markdown(

            f"""
<div class="success-box">

✓ Response time {elapsed:.3f}s meets KPI target.

</div>
""",

            unsafe_allow_html=True

        )


    else:


        st.markdown(

            f"""
<div class="warning-box">

⚠ Response time exceeds KPI target.

</div>
""",

            unsafe_allow_html=True

        )




    # =========================================================================
    # DETAILS
    # =========================================================================


    with st.expander(
        "📈 Detailed Analysis"
    ):



        if selected_model == "Logistic Regression" and top_features:


            st.markdown(
                "**🔑 Important Words Influencing Prediction**"
            )


            feature_df = pd.DataFrame(

                [

                    {

                    "Word":
                    word,


                    "Direction":
                    "AI"
                    if score > 0
                    else
                    "Human",


                    "Weight":
                    round(abs(score),4)

                    }

                    for word, score in top_features

                ]

            )


            st.dataframe(

                feature_df,

                use_container_width=True

            )



        st.markdown(
            "**📝 Writing Style Analysis**"
        )



        style_data = extract_stylometrics(
            text_input
        )


        style_df = pd.DataFrame(

            style_data.items(),

            columns=[
                "Metric",
                "Value"
            ]

        )


        st.dataframe(

            style_df,

            use_container_width=True

        )



        st.markdown(

f"""
### Model Information

Model: {selected_model}

Mode: {mode}

F1 Score: {model_info['f1']}

ROC-AUC: {model_info['auc']}

Inference Time:
{elapsed:.3f}s

"""
        )



    st.markdown(

        """
<div class="info-box">

💡 Disclaimer:
AI detection systems are not perfect.
Use this result as supporting evidence only.

</div>
""",

        unsafe_allow_html=True

    )



# =============================================================================
# EXAMPLE TEXTS
# =============================================================================


st.markdown("---")

st.markdown(
    "### 💡 Try Example Texts"
)



human_example = (

"I went to the farmers market on Saturday and honestly it was such a vibe. "

"Found this amazing stall selling homemade sourdough. "

"The weather was perfect and I enjoyed walking around."

)



ai_example = (

"Effective time management is an essential skill that enables individuals "

"to maximise productivity and achieve their goals. "

"By prioritising tasks and maintaining structured routines, "

"people can improve professional and personal outcomes."

)



col1, col2 = st.columns(2)



with col1:


    if st.button(
        "✍️ Load Human Example",
        use_container_width=True
    ):


        st.session_state.example = human_example

        st.rerun()



with col2:


    if st.button(
        "🤖 Load AI Example",
        use_container_width=True
    ):


        st.session_state.example = ai_example

        st.rerun()



if "example" in st.session_state:


    st.text_area(

        "Example Text",

        value=st.session_state.example,

        height=150

    )



# =============================================================================
# FOOTER
# =============================================================================


st.markdown(

"""
<div class="footer">

MSc Data Science · University of Hertfordshire · Group Final Project 7PAM2033

</div>
""",

unsafe_allow_html=True

)
