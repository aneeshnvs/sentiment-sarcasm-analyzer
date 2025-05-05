# src/test_preprocess_sentiment.py

from preprocess_sentiment import preprocess_sentiment_data

# Call the function with paths to your train/test CSV files
X_train, y_train, X_test, y_test, tokenizer = preprocess_sentiment_data(
    'data/sentiment/train.csv',
    'data/sentiment/test.csv'
)

print("✅ Preprocessing successful!")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
