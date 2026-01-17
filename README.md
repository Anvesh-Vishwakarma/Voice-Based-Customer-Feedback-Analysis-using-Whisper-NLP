# 🎙️ Voice-Based Sentiment Analysis System

An end-to-end **voice sentiment analysis application** that records live audio input, converts speech to text using OpenAI Whisper, predicts sentiment using a trained Machine Learning model, and stores results in a SQLite database.

This project demonstrates real-world skills in **audio processing, speech recognition, NLP, ML inference, and data persistence**.

---

## 📌 Key Features

- 🎤 **Live Audio Recording** using microphone input  
- 🧠 **Speech-to-Text** conversion using **OpenAI Whisper**
- 📊 **Sentiment Classification** (Negative / Neutral / Positive)
- 🧹 **Text Cleaning & Preprocessing**
- 🗄️ **SQLite Database Storage** for transcripts and predictions
- 🧩 Modular, production-ready Python code

---

## 🏗️ Project Architecture

```bash
├── main.py                     # Main application entry point
├── models/
│   ├── sentiment_model.pkl     # Trained sentiment classification model
│   └── vectorizer.pkl          # Text vectorizer
├── output.wav                  # Recorded audio file
├── voice_sentiment.db          # SQLite database
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```


---

## 🧠 Tech Stack

| Category | Technology |
|--------|------------|
| Language | Python |
| Speech Recognition | OpenAI Whisper |
| Audio Processing | sounddevice, scipy |
| NLP & ML | Scikit-learn |
| Model Persistence | joblib |
| Database | SQLite |
| Utilities | regex, datetime |

---

## 🚀 How It Works

1. User presses **Enter** to start recording
2. Audio is captured via microphone
3. Whisper converts speech to text
4. Text is cleaned and vectorized
5. ML model predicts sentiment
6. Result is stored in SQLite database

---

## ▶️ Running the Project

### 1️ Install Dependencies

```
pip install -r requirements.txt
```

### 2 Run the Application
```
python main.py
```
### 3️ Controls

* Press Enter → Start Recording
- say anything like: Bad flight, not good service or good crew members

* Press Enter again → Stop Recording
- It will generate a sentiment- Negative, Positive or Neutral
- Then all the data like id, sentiment, time will be sent to DB

