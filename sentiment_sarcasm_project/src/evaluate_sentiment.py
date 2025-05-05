from preprocess_sentiment import preprocess_sentiment_data
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load preprocessed data
X_train, y_train, X_test, y_test, tokenizer = preprocess_sentiment_data(
    'data/sentiment/train.csv',
    'data/sentiment/test.csv',
    max_vocab=10000,
    max_len=100
)

# Load trained model
model = load_model('models/sentiment_model.keras')

# Predict (returns probability between 0 and 1)
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Evaluate
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

print("🧱 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
