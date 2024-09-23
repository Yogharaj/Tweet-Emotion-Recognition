import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from azureml.core.model import Model

# Load your tokenizer
def load_tokenizer():
    import joblib
    tokenizer_path = Model.get_model_path("tweet_tokenizer")  # Assuming you saved it during training
    tokenizer = joblib.load(tokenizer_path)
    return tokenizer

def init():
    global model
    global tokenizer
    model_name = "tweet_sentiment_model"
    model_path = Model.get_model_path(model_name)
    model = tf.keras.models.load_model(model_path)
    tokenizer = load_tokenizer()

def run(raw_data):
    try:
        data = json.loads(raw_data)
        tweet = data['content']
        
        # Preprocess the tweet
        processed_input = preprocess_tweet(tweet)
        
        # Make the prediction
        prediction = model.predict(processed_input)
        result = {
            "prediction": int(np.argmax(prediction)),
            "confidence": float(np.max(prediction))
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})

def preprocess_tweet(tweet):
    # Clean the tweet (same function as in training)
    import re
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'\@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.lower().strip()
    
    # Tokenize and pad the tweet
    sequences = tokenizer.texts_to_sequences([tweet])
    padded_sequence = pad_sequences(sequences, maxlen=100, padding='post')
    
    return padded_sequence
