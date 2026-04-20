import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text
from config import MODEL_PATH, TOKENIZER_PATH, MAX_SEQUENCE_LENGTH

def test_single_email():
    """
    Manually test the trained AI model with custom email text.
    """
    print("\n" + "="*50)
    print("🔍 PHISHING SHIELD: MANUAL TEST MODE")
    print("="*50)

    # 1. Load the AI assets
    try:
        print(f"📂 Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        
        print(f"📖 Loading tokenizer from: {TOKENIZER_PATH}")
        tokenizer = joblib.load(TOKENIZER_PATH)
        print("✅ Assets loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error: Could not load model or tokenizer. {e}")
        return

    while True:
        print("-" * 50)
        email_input = input("Enter Email Content to Scan (or type 'exit' to quit): \n> ")
        
        if email_input.lower() == 'exit':
            print("👋 Exiting Test Mode. Happy Coding!")
            break

        if not email_input.strip():
            continue

        # 2. Pre-process the input (Same logic used in training)
        cleaned = clean_text(email_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        # 3. Perform AI Inference
        prediction_prob = float(model.predict(padded, verbose=0)[0][0])
        
        # 4. Display Results
        print("\n--- SCAN REPORT ---")
        print(f"Cleaned Text: {cleaned[:100]}...")
        print(f"AI Score: {prediction_prob:.4f} (Probability of being Phishing)")
        
        if prediction_prob > 0.85:
            print("Verdict: 🚨 CRITICAL THREAT (Highly likely Phishing)")
        elif prediction_prob > 0.5:
            print("Verdict: ⚠️ HIGH RISK (Suspected Phishing)")
        elif prediction_prob > 0.2:
            print("Verdict: 🟡 MEDIUM RISK (Check links carefully)")
        else:
            print("Verdict: ✅ SAFE (Legitimate content)")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    test_single_email()