import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove links
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # remove @mentions
    text = re.sub(r"#", "", text)               # remove '#' but keep hashtag word
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # remove punctuation/numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Main function to preprocess sentiment data
def preprocess_sentiment_data(train_path, test_path, max_vocab=10000, max_len=100):
    # Load datasets
    train_df = pd.read_csv(train_path, encoding='latin1')
    test_df = pd.read_csv(test_path, encoding='latin1')

    # Keep only positive and negative sentiments
    train_df = train_df[train_df['sentiment'].isin(['positive', 'negative'])]
    test_df = test_df[test_df['sentiment'].isin(['positive', 'negative'])]

    # Map sentiments to 0/1
    sentiment_map = {'negative': 0, 'positive': 1}
    train_df['sentiment'] = train_df['sentiment'].map(sentiment_map)
    test_df['sentiment'] = test_df['sentiment'].map(sentiment_map)

    # Drop any rows where mapping failed (to remove NaN)
    train_df = train_df.dropna(subset=['sentiment'])
    test_df = test_df.dropna(subset=['sentiment'])

    # Ensure sentiment is integer
    train_df['sentiment'] = train_df['sentiment'].astype(int)
    test_df['sentiment'] = test_df['sentiment'].astype(int)

    # Clean the text
    train_df['text'] = train_df['text'].astype(str).apply(clean_text)
    test_df['text'] = test_df['text'].astype(str).apply(clean_text)

    # Tokenization
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['text'])

    # Text to sequences
    X_train = tokenizer.texts_to_sequences(train_df['text'])
    X_test = tokenizer.texts_to_sequences(test_df['text'])

    # Padding
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')

    # Extract labels
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values

    return X_train, y_train, X_test, y_test, tokenizer
