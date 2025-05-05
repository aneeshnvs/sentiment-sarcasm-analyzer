# Sentiment + Sarcasm Detection Web App

This project is a real-time, deep learning-based **Sentiment and Sarcasm Detection Web Application**, developed to showcase Natural Language Processing (NLP) capabilities using Python, Keras, Streamlit, and supporting ML libraries. It allows users to input any text (e.g., a movie review), and receive:

* The **predicted sentiment** (Positive or Negative)
* A **sarcasm flag** if the tone of the text contradicts the literal sentiment
* Confidence scores and raw prediction scores for transparency

---

## Why This Project Was Built

This app was designed to solve a key problem in NLP: **sentiment models often misinterpret sarcasm**. For example, the phrase:

> "Oh great, another amazing sequel... said no one ever."

contains positive words but negative intent. The goal was to:

* Build two separate models for **literal sentiment** and **sarcasm detection**
* Integrate them into a single pipeline for smarter interpretation
* Deploy it via a clean, demo-ready web interface
* Make it a portfolio-quality project for interviews, GitHub, and LinkedIn

---

## Key Features

* LSTM-based **Sentiment Classifier** trained on real-world reviews
*  LSTM-based **Sarcasm Detector** trained on headline sarcasm data
*  Web interface built with **Streamlit**
*  Detects and **flags sarcasm** before showing sentiment
*  Shows **confidence scores** and raw logits for better insight
*  Modular code structure with preprocessing, model training, and app layers

---

## Tech Stack

* **Programming Language**: Python 3.10
* **Machine Learning**: TensorFlow / Keras, scikit-learn
* **NLP Tools**: NLTK, regex, stopword removal, tokenization
* **Web Framework**: Streamlit
* **Deployment Ready**: Streamlit Cloud or Hugging Face Spaces

---

## Folder Structure

```
sentiment_project/
├── data/
│   ├── sentiment/
│   │   ├── train.csv
│   │   └── test.csv
│   └── sarcasm/
│       └── Sarcasm_Headlines_Dataset.json
├── models/
│   ├── sentiment_model.keras
│   └── sarcasm_model.keras
├── src/
│   ├── app.py
│   ├── preprocess_sentiment.py
│   ├── preprocess_sarcasm.py
│   ├── train_sentiment.py
│   └── train_sarcasm.py
└── README.md
```

---

## How It Works

### Step 1: Preprocessing

* Cleans text (lowercase, removes URLs/usernames/punctuation)
* Removes stopwords using NLTK
* Applies tokenization and padding

### Step 2: Model Training

* **Sentiment model**: LSTM trained on labeled review data
* **Sarcasm model**: LSTM trained on sarcasm-labeled headlines

### Step 3: Inference in Streamlit App

* User submits text
* App predicts sarcasm first
* If sarcastic: flags result and warns user
* Then predicts literal sentiment
* Displays both predictions with confidence

---

## 🔧 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/aneeshnvs/sentiment-sarcasm-analyzer.git
cd sentiment-sarcasm-analyzer
```

### 2. Setup Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web App

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example Inputs to Try

| Input Text                                         | Expected Result                                     |
| -------------------------------------------------- | --------------------------------------------------- |
| "Totally loved it. Cried twice!"                   | Sentiment: Positive                                 |
| "Yeah, that was soooo entertaining... 🙄"          | Sentiment: Positive, **Sarcasm Detected**           |
| "Worst movie ever. Don't waste your time."         | Sentiment: Negative                                 |
| "Loved the acting. Especially the cardboard hero." | **Sarcasm Detected**, Sentiment: Positive (flagged) |

---

## Future Work

* Save tokenizers as `.pkl` for faster load
* Add emoji/sentiment visualizations in UI
* Deploy on Streamlit Cloud with public URL
* Fine-tune sarcasm model on IMDb/Reddit sarcastic reviews
* Replace LSTM with BERT for deeper context

---

## 👤 Author

**Aneesh Nalluri**
☙ Charlotte, NC
🔗 [LinkedIn](https://www.linkedin.com/in/aneeshnvs/)
📧 Contact via GitHub or LinkedIn

---

