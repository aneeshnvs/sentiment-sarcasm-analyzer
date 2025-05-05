import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Clean the text (same as in preprocessing)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# 2. Load model and tokenizer
model = load_model('models/sentiment_model.keras')

# If you haven’t saved your tokenizer yet:
from preprocess_sentiment import preprocess_sentiment_data
_, _, _, _, tokenizer = preprocess_sentiment_data(
    'data/sentiment/train.csv',
    'data/sentiment/test.csv'
)

# 3. Prediction function
def predict_sentiment(text, max_len=100):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prob = model.predict(padded)[0][0]

    label = "Positive" if prob > 0.5 else "Negative"
    confidence = round(prob if prob > 0.5 else 1 - prob, 2)

    print(f"🔮 Prediction: {label} ({confidence * 100:.1f}% confidence)")
    return label, confidence


if __name__ == "__main__":
    predict_sentiment("I absolutely loved the way this worked!")
    predict_sentiment("This was the worst experience ever.")
    predict_sentiment("It’s okay, nothing special.")