from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the model
model = load_model('text_summarization_rnn_light.h5')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the tokenizers
with open('text_tokenizer.pickle', 'rb') as handle:
    text_tokenizer = pickle.load(handle)

with open('summary_tokenizer.pickle', 'rb') as handle:
    summary_tokenizer = pickle.load(handle)

max_text_len = 300
max_summary_len = 50

def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

def generate_summary(text):
    text_sequence = text_tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=max_text_len, padding='post')

    summary_sequence = np.zeros((1, max_summary_len, 1))

    prediction = model.predict([text_sequence, summary_sequence], verbose=0)
    summary_tokens = np.argmax(prediction, axis=-1).flatten()

    summary = ' '.join([summary_tokenizer.index_word[token] for token in summary_tokens if token != 0])
    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        article_text = scrape_article(url)
        preprocessed_text = preprocess_text(article_text)
        summary = generate_summary(preprocessed_text)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
