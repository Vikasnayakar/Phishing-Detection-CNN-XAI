# Phishing Detection Framework using CNN and XAI

### Project Overview
This project focuses on the detection of phishing emails using a Convolutional Neural Network (CNN). To bridge the gap between AI decision-making and user trust, we integrate Explainable AI (XAI). The system not only classifies an email as safe or phishing but also highlights specific suspicious features (like keywords or URLs) to educate the user on why a particular decision was made.

![UI/UX Dashboard Preview](assets/dashboard_preview.png)

### Project Team and Guide
* Team Members: Praveen B K, Gireesh Patil, Bharghav, M Vishwanth
* Guide: Dr. P Karthikeyan
* Institution: School of Computer Science & Engineering, RV University, Bengaluru

### Tech Stack
* Language: Python 3.10+
* Backend Framework: FastAPI (Uvicorn)
* Deep Learning: TensorFlow, Keras (1D-CNN architecture)
* Preprocessing: NLTK, Scikit-learn
* Frontend: HTML5, CSS3, JavaScript (Real-time XAI Dashboard)

### Repository Structure
* /backend: Contains the FastAPI server, model training scripts (train_model.py), and the saved tokenizer.
* /frontend: The web interface for users to paste email content and view analysis.
* requirements.txt: List of dependencies required to run the project.

### Instructions to Run
1. Clone the repository:
   git clone https://github.com/Vikasnayakar/Phishing-Detection-CNN-XAI.git
   cd Phishing-Detection-CNN-XAI

2. Create a virtual environment:
   python -m venv venv
   source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Start the Backend:
   uvicorn backend.main:app --reload

5. Open the Frontend:
   Open frontend/index.html in a web browser.
