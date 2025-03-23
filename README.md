# Sentiment Analysis Web App

## Overview
This project consists of a sentiment analysis web application built using **Streamlit** for the frontend and **Flask** for the backend. The model used is **cardiffnlp/twitter-roberta-base-sentiment-latest**, which classifies input sentences into three sentiment categories: **Positive, Neutral, and Negative**.

## Features
- Users can input multiple sentences for analysis.
- Sentiment analysis is performed using a pre-trained **RoBERTa** model.
- Sentiment percentages are calculated and displayed.
- A **Flask API** endpoint (`/sentiment`) is available for programmatic access.
- **Live Web App:** [Sentiment Analysis Web App](https://huggingface.co/spaces/Tahir5/Sentiment-Analysis)

---

## Project Structure
```
📂 sentiment-analysis
│── .gitignore  # Ignored files
│── app.py  # Flask API for sentiment analysis
│── stream.py  # Streamlit app for user interaction
│── requirements.txt  # Dependencies
│── README.md  # Project documentation
```

---

## Installation and Setup
### 1. Clone the Repository
```bash
git clone <repo-url>
cd sentiment-analysis
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Flask API (Backend)
### 1. Run the Flask App
```bash
python app.py
```
### 2. API Endpoint
Once the server is running, the API can be accessed at:
```
POST http://localhost:5000/sentiment
```
**Request Body (JSON Example):**
```json
{
    "sentence1": "I love this product!",
    "sentence2": "It's okay, nothing special.",
    "sentence3": "I hate waiting in long lines."
}
```
**Response Example:**
```json
{
    "positive": 33.33,
    "neutral": 33.33,
    "negative": 33.33
}
```

---

## Running the Streamlit Web App (Frontend)
### 1. Run the Streamlit App
```bash
streamlit run stream.py
```
The web app will be accessible at:  
**http://localhost:8501**

---

## Technologies Used
- **Python** (Backend and Frontend)
- **Flask** (API Development)
- **Streamlit** (Web Interface)
- **Transformers** (Hugging Face model)
- **NumPy, SciPy** (Mathematical Operations)
- **Collections (Counter)** (Sentiment Analysis Computation)

---

## Future Enhancements
- Deploy the application on **AWS/GCP/Heroku**.
- Improve UI/UX with interactive graphs.
- Support for more languages.

---

## License
This project is open-source and available under the **MIT License**.

---

## Contact
For queries or contributions, reach out to:
**Tahir Abbas Shaikh**  
📧 tahirabbasshaikh555@gmail.com  
📞 +923022024206
