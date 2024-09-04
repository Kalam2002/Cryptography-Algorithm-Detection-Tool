from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import math
from collections import Counter
from Crypto.Cipher import AES, DES, Blowfish
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import hashlib
import base64
import rsa

app = Flask(__name__)

def calculate_entropy(text):
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum([p * math.log2(p) for p in prob])
    return entropy

def extract_features(ciphertext):
    counter = Counter(ciphertext)
    char_freq = [counter.get(chr(i), 0) for i in range(256)]  # ASCII frequencies
    char_freq = np.array(char_freq) / len(ciphertext)  # Normalize by length
    entropy = calculate_entropy(ciphertext)
    features = np.append(char_freq, entropy)
    return features

def predict_algorithm(ciphertext):
    model = joblib.load('crypto_classifier_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = extract_features(ciphertext)
    features_scaled = scaler.transform([features])
    predicted_algo = model.predict(features_scaled)
    algo_dict = {0: 'AES', 1: 'DES', 2: 'Blowfish', 3: 'RSA', 4: 'SHA-256', 5: 'MD5', 6: 'Caesar Cipher'}
    return algo_dict.get(predicted_algo[0], 'Unknown')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ciphertext = data.get('ciphertext', '')
    result = predict_algorithm(ciphertext)
    return jsonify({'algorithm': result})

if __name__ == '__main__':
    app.run(debug=True)
