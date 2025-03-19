import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from collections import Counter

# Load model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyze_sentiment_for_sentences(sentences):
    sentiments = []
    for sentence in sentences:
        encoded_input = tokenizer(sentence, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        highest_sentiment_idx = np.argmax(scores)
        highest_sentiment_label = config.id2label[highest_sentiment_idx]
        sentiments.append(highest_sentiment_label)
    return sentiments

def calculate_sentiment_percentages(sentiments):
    sentiment_counts = Counter(sentiments)
    total_sentiments = len(sentiments)
    sentiment_percentages = {"😊 positive": 0, "😐 neutral": 0, "😔 negative": 0}
    for sentiment, count in sentiment_counts.items():
        if sentiment == "positive":
            sentiment_percentages["😊 positive"] = (count / total_sentiments) * 100
        elif sentiment == "neutral":
            sentiment_percentages["😐 neutral"] = (count / total_sentiments) * 100
        else:
            sentiment_percentages["😔 negative"] = (count / total_sentiments) * 100
    return sentiment_percentages

# Streamlit UI
st.title("✨ Sentiment Analysis Web App")
st.write("Enter sentences below to analyze their sentiment.")

# User input
user_input = st.text_area("Enter sentences (one per line):")
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentences = user_input.split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
        sentiments = analyze_sentiment_for_sentences(sentences)
        sentiment_percentages = calculate_sentiment_percentages(sentiments)
        
        st.subheader("📊 Sentiment Analysis Results")
        st.write(sentiment_percentages)
    else:
        st.warning("⚠️ Please enter at least one sentence.")
