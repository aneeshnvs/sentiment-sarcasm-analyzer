import pandas as pd

train_df = pd.read_csv("data/sentiment/train.csv", encoding="latin1")
print("Sentiment value counts in training data:")
print(train_df['sentiment'].value_counts())
