from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text

app = FastAPI(
    title="Phishing Shield Pro API", 
    description="Enhanced Hybrid AI Security System",
    version="3.0.0"
)

# --- SECURITY: CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONSTANTS ---
MODEL_PATH = "backend/models/cnn_phishing_model.h5"
TOKENIZER_PATH = "backend/models/tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 300 

# LAYER 1: DOMAIN BLACKLIST
BLACKLIST_DOMAINS = ["scam-verify.net", "login-microsoft.live", "bank-update-portal.biz", "karnataka-bank-verify.in"]

model = None
tokenizer = None

@app.on_event("startup")
def load_assets():
    global model, tokenizer
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        tokenizer = joblib.load(TOKENIZER_PATH)
        print(f"[{datetime.now()}] ✅ SYSTEM READY: Hybrid XAI Mode Active.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load assets: {str(e)}")

# --- DATA MODELS ---
class EmailRequest(BaseModel):
    text: str

class ScanResponse(BaseModel):
    verdict: str
    confidence_score: float
    threat_level: str
    scan_time: str
    triggers: list  # For Explainable AI Highlighting
    is_blacklisted: bool # For Blacklist check status

# --- API ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "online", "features": ["CNN", "Heuristics", "Blacklist", "XAI"]}

@app.post("/predict", response_model=ScanResponse)
async def predict(request: EmailRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="AI Model Offline.")

    start_time = time.time()
    
    try:
        # 1. RAW DATA & BLACKLIST CHECK
        raw_text = request.text
        is_blacklisted = any(domain in raw_text.lower() for domain in BLACKLIST_DOMAINS)
        
        # 2. PRE-PROCESS (Ensure utils.py clean_text supports Unicode for Kannada)
        cleaned = clean_text(raw_text).lower()
        
        # 3. AI PREDICTION
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        prediction_prob = float(model.predict(padded)[0][0])
        
        # 4. EXPLAINABLE AI (XAI) & HEURISTIC TRIGGERS
        # Combined English and Kannada keywords
        trigger_words = ['urgent', 'verify', 'password', 'payment', 'suspended', 'login', 'quick task', 'ತುರ್ತು', 'ಹಣ']
        financial_words = ['gift card', 'google play', 'amazon card', 'wire transfer', 'reimburse', 'buy']
        
        # Find exactly which triggers exist in the text to send back to UI
        found_triggers = [word for word in (trigger_words + financial_words) if word in cleaned]
        
        has_urgent_word = any(word in cleaned for word in trigger_words)
        has_link = "http" in cleaned
        has_financial_request = any(word in cleaned for word in financial_words)

        # 5. FINAL VERDICT LOGIC (Layered Security)
        if prediction_prob > 0.50 or is_blacklisted or (has_urgent_word and (has_link or has_financial_request)):
            verdict = "🚨 PHISHING"
            threat = "HIGH"
            # If blacklisted, we force a near-100% score
            display_prob = 0.9999 if is_blacklisted else max(prediction_prob, 0.95)
        elif prediction_prob > 0.10 or has_link or has_financial_request or len(found_triggers) > 0:
            verdict = "⚠️ SUSPICIOUS"
            threat = "MEDIUM"
            display_prob = max(prediction_prob, 0.45)
        else:
            verdict = "✅ SAFE"
            threat = "LOW"
            display_prob = prediction_prob

        execution_time = f"{time.time() - start_time:.4f}s"

        # LOGGING TO TERMINAL FOR DEBUGGING
        print(f"\n{'='*40}")
        print(f"📊 VERDICT: {verdict}")
        print(f"🎯 AI PROB: {prediction_prob * 100:.2f}%")
        print(f"🔍 TRIGGERS: {found_triggers}")
        print(f"🚫 BLACKLISTED: {is_blacklisted}")
        print(f"{'='*40}\n")

        return {
            "verdict": verdict,
            "confidence_score": round(display_prob, 4),
            "threat_level": threat,
            "scan_time": execution_time,
            "triggers": found_triggers,
            "is_blacklisted": is_blacklisted
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)