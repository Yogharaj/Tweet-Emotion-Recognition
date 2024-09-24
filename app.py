from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import joblib
import pandas as pd
import os
df = pd.read_csv('tweet_emotions.csv')
labels = df['sentiment'].unique()
labels = sorted(labels) 
label_to_index = {label: idx for idx, label in enumerate(labels)}
index_to_label = {idx: label for idx, label in enumerate(labels)}
app = Flask(__name__)
MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "tweet_sentiment_model.h5")
model = tf.keras.models.load_model(model_path)
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts([])

def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'\@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.lower().strip()
    sequences = tokenizer.texts_to_sequences([tweet])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return padded_sequence

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        tweet = data['content']
        processed_input = preprocess_tweet(tweet)
        prediction = model.predict(processed_input)
        predicted_label_index = np.argmax(prediction)
        predicted_label = index_to_label[predicted_label_index]
        
        result = {
            "prediction": predicted_label,
            "confidence": float(np.max(prediction))
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/test")
def fun():
    return "Hello world"

@app.route("/")
def fun1():
    return "Home page"

@app.route("/testendpoint")
def fun2():
    return "Test working"

if __name__ == "__main__":
    app.run(host="0.0.0.0")
