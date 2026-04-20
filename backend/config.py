import os

# ==========================================
# 🧠 AI & TEXT PROCESSING HYPERPARAMETERS
# ==========================================
# Number of unique words the AI will remember
MAX_WORDS = 15000 

# Max length of an email (Longer emails are cut, shorter are padded)
MAX_SEQUENCE_LENGTH = 300 

# The size of the vector space for word meanings
EMBEDDING_DIM = 128 


# ==========================================
# 🚀 TRAINING SETTINGS
# ==========================================
BATCH_SIZE = 64        # Number of samples processed before updating weights
EPOCHS = 10            # Number of times the AI sees the entire dataset
VALIDATION_SPLIT = 0.2 # 20% of data used for testing, 80% for training
LEARNING_RATE = 0.001  # Initial speed of the AI's learning


# ==========================================
# 📁 FILE & FOLDER PATHS
# ==========================================
# Using absolute paths or joining paths ensures VS Code finds them correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your 82k email dataset
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "emails.csv")

# Folder where the AI's 'Brain' and 'Dictionary' will live
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True) # Automatically creates the folder if missing

MODEL_PATH = os.path.join(MODEL_DIR, "cnn_phishing_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")


# ==========================================
# 🛡️ SYSTEM METADATA
# ==========================================
PROJECT_NAME = "Phishing Shield AI"
VERSION = "2.0.1"
THRESHOLD = 0.5 # Anything above this 0.5 probability is flagged as Phishing