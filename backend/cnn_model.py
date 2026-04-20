import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization

def build_cnn_model(vocab_size, max_length, embedding_dim=128):
    """
    Constructs a 1D Convolutional Neural Network for Text Classification.
    Features:
    - Embedding Layer: Learns semantic relationships between words.
    - Conv1D: Scans word patterns (n-grams).
    - GlobalMaxPooling: Captures the most important features.
    - Dropout & BatchNormalization: Prevents overfitting (the 80% val accuracy issue).
    """
    
    model = Sequential([
        # 1. Word Embedding Layer
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        
        # 2. Convolutional Layer (Scanning 5 words at a time)
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(), # Stabilizes training
        
        # 3. Pooling Layer
        GlobalMaxPooling1D(),
        
        # 4. Fully Connected Layers
        Dense(128, activation='relu'),
        Dropout(0.4), # Reduced slightly to balance learning and memorization
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # 5. Output Layer (Sigmoid for binary 0/1 classification)
        Dense(1, activation='sigmoid')
    ])

    # Optimization Setup
    # Using 'adam' as it's the industry standard for text classification
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    print("✅ CNN Architecture built successfully.")
    model.summary() # Logs the layer structure to your terminal
    
    return model

def save_trained_model(model, path):
    """Saves the model in the recommended native Keras format."""
    model.save(path)
    print(f"📁 Model saved to: {path}")

def load_existing_model(path):
    """Loads a previously trained model."""
    try:
        model = load_model(path)
        print(f"🧠 Model loaded successfully from {path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None