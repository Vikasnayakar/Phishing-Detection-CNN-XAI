import pandas as pd
import numpy as np
import re
import joblib
import os
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_WORDS, MAX_SEQUENCE_LENGTH, TOKENIZER_PATH, DATA_PATH

# Setup logging for better debugging in VS Code terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """
    Advanced text cleaning:
    - Preserves URL patterns (critical for phishing detection)
    - Retains numbers (urgent deadlines like 24h, 48h)
    - Removes HTML noise
    - Standardizes whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Standardize to lowercase
    text = text.lower()
    
    # 2. Remove HTML tags but leave a space to prevent merging words
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Handle URLs: Instead of deleting them, we normalize them
    # This helps the AI recognize the structure of a link
    text = re.sub(r'http\S+|www\S+', ' http_link ', text)
    
    # 4. Filter: Keep only alphanumeric characters and spaces
    # Keeping numbers is vital for "Urgency" detection
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 5. Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'[^a-zA-Z0-9\s\u0C80-\u0CFF]', '', text)
    return text

def get_email_metadata(text):
    """
    Extra feature: Extracts metadata to show on your HTML frontend.
    """
    return {
        "length": len(text),
        "has_urgency_words": any(word in text.lower() for word in ['urgent', 'limited', 'suspended', 'action']),
        "link_count": text.lower().count("http")
    }

def load_and_preprocess_data():
    """
    Loads the 82k dataset, fits the tokenizer, and saves the dictionary.
    Includes data validation checks.
    """
    if not os.path.exists(DATA_PATH):
        logging.error(f"DATASET MISSING: Check {DATA_PATH}")
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")

    logging.info(f"📂 Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop rows with missing values to prevent training errors
    df.dropna(subset=['text', 'label'], inplace=True)

    if 'text' not in df.columns or 'label' not in df.columns:
        logging.error("CSV structure invalid. Columns 'text' and 'label' are required.")
        raise ValueError("The CSV must contain 'text' and 'label' columns.")

    logging.info(f"🧹 Cleaning {len(df)} emails... this might take a moment.")
    df['clean_text'] = df['text'].apply(clean_text)

    # Initialize and Fit Tokenizer
    logging.info("📖 Building word dictionary (Tokenizer)...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_text'])
    
    # Ensure the models directory exists before saving
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    joblib.dump(tokenizer, TOKENIZER_PATH)
    logging.info(f"✅ Tokenizer saved successfully to {TOKENIZER_PATH}")

    # Convert text to padded numeric sequences
    logging.info("🔢 Converting text to numeric sequences...")
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    y = df['label'].values

    return X, y, tokenizer

def preprocess_single_email(text, tokenizer):
    """
    Helper function for main.py to process a single user input.
    """
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded