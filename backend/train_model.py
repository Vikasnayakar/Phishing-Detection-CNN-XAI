import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils import load_and_preprocess_data
from cnn_model import build_cnn_model
from config import MAX_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, MODEL_PATH

def train():
    # 1. Load data using the improved utils logic (keeps numbers/links)
    print("📦 Loading and preprocessing dataset...")
    X, y, tokenizer = load_and_preprocess_data()
    
    # 2. Split into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    # 3. Calculate Class Weights
    # This forces the model to pay equal attention to Phishing and Safe emails
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))

    # 4. Initialize the CNN Architecture
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
    print(f"🧠 Building CNN with vocab size: {vocab_size}")
    model = build_cnn_model(vocab_size, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

    # 5. Advanced Callbacks for Better Training
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    callbacks = [
        # Stop training if the model stops improving (prevents overfitting)
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        
        # Save the absolute best version of the brain found during training
        ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        
        # Lower the learning rate if training plateaus (finer adjustments)
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
    ]

    # 6. Start the Training Process
    print(f"🚀 Training started for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict, # Apply weights
        callbacks=callbacks
    )

    # 7. Final Evaluation
    print("\n📊 Evaluating model on unseen test data...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"✅ Final Test Accuracy: {accuracy * 100:.2f}%")
    print(f"📁 Model saved as: {MODEL_PATH}")

if __name__ == "__main__":
    train()