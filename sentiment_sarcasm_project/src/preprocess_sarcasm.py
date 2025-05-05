import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Reuse the clean_text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_sarcasm_data(json_path, max_vocab=10000, max_len=100):
    # Load JSON lines
    with open(json_path, 'r') as file:
        data = [json.loads(line) for line in file]

    df = pd.DataFrame(data)
    df = df[['headline', 'is_sarcastic']]
    df.dropna(inplace=True)

    # Clean text
    df['headline'] = df['headline'].astype(str).apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['headline'])

    # Convert to padded sequences
    sequences = tokenizer.texts_to_sequences(df['headline'])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    X = padded
    y = df['is_sarcastic'].values

    return X, y, tokenizer
