from preprocess_sarcasm import preprocess_sarcasm_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

# Set paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '../data/sarcasm/Sarcasm_Headlines_Dataset.json')

# Load data
X, y, tokenizer = preprocess_sarcasm_data(DATA_PATH)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=5,
          batch_size=64,
          callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])

# Save model
MODEL_PATH = os.path.join(BASE_DIR, '../models/sarcasm_model.keras')
model.save(MODEL_PATH)

print("✅ Sarcasm model saved successfully.")
