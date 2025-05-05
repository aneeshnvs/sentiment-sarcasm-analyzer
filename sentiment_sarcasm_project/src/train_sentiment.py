from preprocess_sentiment import preprocess_sentiment_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Load preprocessed data
X_train, y_train, X_test, y_test, tokenizer = preprocess_sentiment_data(
    'data/sentiment/train.csv',
    'data/sentiment/test.csv',
    max_vocab=10000,
    max_len=100
)

# Build the Bidirectional LSTM model for binary classification
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary output: positive vs negative
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Callbacks to stop early and reduce learning rate on plateau
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    callbacks=[early_stop, reduce_lr]
)

# Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/sentiment_model.keras")
print("✅ Sentiment model saved to models/sentiment_model.keras")
