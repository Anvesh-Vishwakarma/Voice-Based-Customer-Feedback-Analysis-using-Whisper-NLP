import sounddevice as sd
from scipy.io.wavfile import write
import keyboard
import time
import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim
import joblib
import os
import re
from datetime import datetime
import sqlite3
from jiwer import wer

# Setupt Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
AUDIO_PATH = os.path.join(BASE_DIR, "output.wav")

# Load Model

model = whisper.load_model("small")
sentiment_model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Creating Methods

def record_audio():

    print("Press enter to start record")
    keyboard.wait("enter")
    print("recording....., press enter again to stop")

    start_time = time.time()

    fs = 16000
    channels = 1
    duration = 10

    # start recording with an arbitary large buffer
    recording = sd.rec(int(fs*duration),samplerate=fs,channels=channels)

    keyboard.wait("enter")
    print("stoping")
    print()

    sd.stop()

    # calculate actual duration 
    duration = time.time() - start_time

    # svaing only the recorded portion
    write("output.wav",fs,recording[:int(duration*fs)])


def speech_to_text():
    
    global model
    #audio_file = open("output.wav","rb")
    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio,n_mels=model.dims.n_mels)

    options = whisper.DecodingOptions()
    decoding_result = whisper.decode(model,mel,options)
    result = decoding_result.text

    # Cleaning the Whisper text
    text = result.lower()
    text = re.sub(r"\b(uh|um|you know|actually|basically)\b", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def predict_sentiment(text):

    #while True:
       # record_audio()
       # output = speech_to_text()
        print()
        print("Customer statement: ",text)

        text_vector = vectorizer.transform([text])
        prediction = sentiment_model.predict(text_vector)[0]
        #prediction_label = vectorizer.inverse_transform([prediction])[0]
        
        print()

        if prediction == 0:
             print("Sentiment: Negative Review")
        elif prediction == 1:
             print("Sentiment: Neutral review")
        else:
             print("Sentiment: Positive Review")

        print()
        

def sentiment_data():
    record_audio()

    transcription = speech_to_text()
    sentiment = predict_sentiment(transcription)

    data = (
        "output.wav",
        transcription,
        sentiment,
        datetime.now()
    )
    return data

def insert_sentiment_record():

    data = sentiment_data()

    conn = None
    try:
        # Connect to DB
        conn = sqlite3.connect("voice_sentiment.db")
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT,
            transcription TEXT,
            sentiment TEXT,
            created_at DATETIME
        )
        """)

        # Insert record
        cursor.execute("""
        INSERT INTO sentiment_analysis 
        (audio_id, transcription, sentiment, created_at)
        VALUES (?, ?, ?, ?)
        """, data)

        conn.commit()
        print("Data inserted successfully")

    except sqlite3.OperationalError as e:
        print("Database operational error:", e)

    except sqlite3.IntegrityError as e:
        print("Data integrity error:", e)

    except Exception as e:
        print("Unexpected error:", e)

    finally:
        if conn:
            conn.close()
            print("Database connection closed")
        

if __name__ == "__main__":
    insert_sentiment_record()


