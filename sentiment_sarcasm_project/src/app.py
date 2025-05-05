import streamlit as st
import os
import nltk
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess_sentiment import clean_text, preprocess_sentiment_data
from preprocess_sarcasm import preprocess_sarcasm_data

# 📥 Ensure stopwords are downloaded
nltk.download('stopwords')

# 📁 Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_SENTIMENT = os.path.join(BASE_DIR, '../models/sentiment_model.keras')
MODEL_SARCASM = os.path.join(BASE_DIR, '../models/sarcasm_model.keras')
TRAIN_PATH = os.path.join(BASE_DIR, '../data/sentiment/train.csv')
TEST_PATH = os.path.join(BASE_DIR, '../data/sentiment/test.csv')

# ✅ Load models
sentiment_model = load_model(MODEL_SENTIMENT)
sarcasm_model = load_model(MODEL_SARCASM)

# ✅ Load tokenizer from sentiment data
_, _, _, _, sentiment_tokenizer = preprocess_sentiment_data(TRAIN_PATH, TEST_PATH)

# ✅ Load tokenizer from sarcasm data
_, _, sarcasm_tokenizer = preprocess_sarcasm_data(os.path.join(BASE_DIR, '../data/sarcasm/Sarcasm_Headlines_Dataset.json'))

# 🎨 Streamlit UI
st.set_page_config(page_title="Smart Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment + Sarcasm Detection App")
st.write("Enter your movie review or comment to check if it's **positive or negative**, and whether it's **sarcastic**.")

user_input = st.text_area("Your Text", height=150)
submit = st.button("Analyze")

# 🔍 Prediction functions
def predict_sentiment(text, max_len=100):
    cleaned = clean_text(text)
    seq = sentiment_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prob = sentiment_model.predict(padded)[0][0]
    label = "Positive" if prob > 0.5 else "Negative"
    confidence = round(prob if prob > 0.5 else 1 - prob, 2)
    return label, confidence

def predict_sarcasm(text, max_len=100):
    cleaned = clean_text(text)
    seq = sarcasm_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prob = sarcasm_model.predict(padded)[0][0]
    is_sarcastic = prob > 0.5
    confidence = round(prob if is_sarcastic else 1 - prob, 2)
    return is_sarcastic, confidence, prob

# 🚀 Run prediction on input
if submit and user_input.strip():
    sentiment_label, sentiment_conf = predict_sentiment(user_input)
    is_sarcastic, sarcasm_conf, raw_prob = predict_sarcasm(user_input)

    st.markdown(f"### 🧠 Sentiment: `{sentiment_label}`")
    st.markdown(f"**Confidence:** {sentiment_conf * 100:.2f}%")

    # 🔍 Show sarcasm result
    if is_sarcastic:
        st.warning(f"⚠️ Sarcasm Detected ({sarcasm_conf * 100:.2f}% confidence)")
        st.markdown("_This review might not reflect literal sentiment due to sarcasm._")
    else:
        st.success(f"✅ No Sarcasm Detected ({sarcasm_conf * 100:.2f}% confidence)")

    # 🧪 Debugging help
    st.text(f"(Raw sarcasm score: {raw_prob:.4f})")
