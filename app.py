from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from collections import Counter

app = Flask(__name__)

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
        ranking = np.argsort(scores)

        highest_sentiment_idx = ranking[-1]
        highest_sentiment_label = config.id2label[highest_sentiment_idx]
        sentiments.append(highest_sentiment_label)

    return sentiments

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    
    # Extract sentences from the JSON object
    sentences = list(data.values())

    # Analyze sentiment for the sentences
    sentiments = analyze_sentiment_for_sentences(sentences)

    # Count the occurrences of each sentiment
    sentiment_counts = Counter(sentiments)

    # Initialize the sentiment percentages with default values of 0 for each sentiment
    sentiment_percentages = {"positive": 0, "neutral": 0, "negative": 0}

    # Calculate the total number of sentiments
    total_sentiments = len(sentiments)

    # Calculate the percentage of each sentiment
    for sentiment, count in sentiment_counts.items():
        sentiment_percentages[sentiment] = (count / total_sentiments) * 100

    return jsonify(sentiment_percentages)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
