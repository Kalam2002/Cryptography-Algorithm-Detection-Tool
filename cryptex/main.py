import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import math
from Crypto.Cipher import AES, DES, Blowfish
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import hashlib
import base64
import rsa  # Import the RSA library

# Step 1: Generate Ciphertext Samples
def generate_ciphertext_samples(num_samples=100):
    """Generate ciphertext samples for AES, DES, Blowfish, RSA, SHA-256, MD5, and Caesar Cipher."""
    plaintext = "This is a test message for encryption algorithm identification."

    samples = []
    labels = []

    for _ in range(num_samples):
        # Generate AES ciphertext
        aes_cipher = generate_aes_ciphertext(plaintext)
        samples.append(aes_cipher)
        labels.append(0)  # Label for AES

        # Generate DES ciphertext
        des_cipher = generate_des_ciphertext(plaintext)
        samples.append(des_cipher)
        labels.append(1)  # Label for DES

        # Generate Blowfish ciphertext
        blowfish_cipher = generate_blowfish_ciphertext(plaintext)
        samples.append(blowfish_cipher)
        labels.append(2)  # Label for Blowfish

        # Generate RSA ciphertext
        rsa_cipher = generate_rsa_ciphertext(plaintext)
        samples.append(rsa_cipher)
        labels.append(3)  # Label for RSA

        # Generate SHA-256 hash
        sha256_hash = generate_sha256_hash(plaintext)
        samples.append(sha256_hash)
        labels.append(4)  # Label for SHA-256

        # Generate MD5 hash
        md5_hash = generate_md5_hash(plaintext)
        samples.append(md5_hash)
        labels.append(5)  # Label for MD5

        # Generate Caesar Cipher ciphertext
        caesar_cipher = generate_caesar_cipher(plaintext, shift=3)
        samples.append(caesar_cipher)
        labels.append(6)  # Label for Caesar Cipher

    return samples, labels

def generate_aes_ciphertext(plaintext):
    key = get_random_bytes(16)  # AES requires a 16-byte key
    cipher = AES.new(key, AES.MODE_ECB)  # Using ECB mode for simplicity
    ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return base64.b64encode(ciphertext).decode()

def generate_des_ciphertext(plaintext):
    key = get_random_bytes(8)  # DES requires an 8-byte key
    cipher = DES.new(key, DES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), DES.block_size))
    return base64.b64encode(ciphertext).decode()

def generate_blowfish_ciphertext(plaintext):
    key = get_random_bytes(16)  # Blowfish key length can vary
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), Blowfish.block_size))
    return base64.b64encode(ciphertext).decode()

def generate_rsa_ciphertext(plaintext):
    """Generate RSA ciphertext."""
    (public_key, private_key) = rsa.newkeys(512)  # Generate RSA public/private keys
    # Truncate plaintext to fit the RSA key size limit
    max_length = (public_key.n.bit_length() + 7) // 8 - 11  # Max size for plaintext with PKCS#1 v1.5 padding
    truncated_plaintext = plaintext[:max_length]  # Ensure plaintext fits within RSA limits
    ciphertext = rsa.encrypt(truncated_plaintext.encode(), public_key)  # Encrypt plaintext with the public key
    return base64.b64encode(ciphertext).decode()

def generate_sha256_hash(plaintext):
    """Generate SHA-256 hash."""
    sha256_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    return sha256_hash

def generate_md5_hash(plaintext):
    """Generate MD5 hash."""
    md5_hash = hashlib.md5(plaintext.encode()).hexdigest()
    return md5_hash

def generate_caesar_cipher(plaintext, shift=3):
    """Generate Caesar Cipher ciphertext with a given shift."""
    ciphertext = ''
    for char in plaintext:
        if char.isalpha():
            shift_base = ord('A') if char.isupper() else ord('a')
            ciphertext += chr((ord(char) - shift_base + shift) % 26 + shift_base)
        else:
            ciphertext += char
    return ciphertext

# Step 2: Feature Extraction
def calculate_entropy(text):
    """Calculate the Shannon entropy of a given text."""
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum([p * math.log2(p) for p in prob])
    return entropy

def extract_features(ciphertext):
    """Extract features like character frequency and entropy from ciphertext."""
    counter = Counter(ciphertext)
    char_freq = [counter.get(chr(i), 0) for i in range(256)]  # ASCII frequencies
    char_freq = np.array(char_freq) / len(ciphertext)  # Normalize by length

    entropy = calculate_entropy(ciphertext)

    features = np.append(char_freq, entropy)
    return features

# Step 3: Train the Machine Learning Model
def train_model(samples, labels):
    """Train a Random Forest classifier on the given samples."""
    # Extract features for each sample
    feature_matrix = np.array([extract_features(sample) for sample in samples])

    # Standardize the feature matrix
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix_scaled, labels, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model and scaler
    joblib.dump(model, 'crypto_classifier_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Step 4: Predict the Algorithm
def predict_algorithm(ciphertext):
    """Predict the encryption algorithm used for the given ciphertext."""
    # Load the trained model and scaler
    model = joblib.load('crypto_classifier_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Extract features from the input ciphertext
    features = extract_features(ciphertext)
    features_scaled = scaler.transform([features])

    # Predict the algorithm
    predicted_algo = model.predict(features_scaled)

    # Map numeric prediction to algorithm names
    algo_dict = {0: 'AES', 1: 'DES', 2: 'Blowfish', 3: 'RSA', 4: 'SHA-256', 5: 'MD5', 6: 'Caesar Cipher'}
    predicted_algo_name = algo_dict.get(predicted_algo[0], 'Unknown')

    return predicted_algo_name

# Main Function
def main():
    # Step 1: Generate samples and labels
    samples, labels = generate_ciphertext_samples()

    # Step 2: Train the model
    train_model(samples, labels)

    # Step 3: Take user input for ciphertext
    user_ciphertext = input("Enter the ciphertext: ")

    # Step 4: Predict the algorithm
    result = predict_algorithm(user_ciphertext)
    print(f"THE PREDICTED CRYPTOGRAPHIC ALGORITHM IS: {result}")

if __name__ == "__main__":
    main()
