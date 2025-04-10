import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from collections import Counter
import pandas as pd
import plotly.express as px

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
    
    sentiment_labels = {
        "positive": "😊 Positive",
        "neutral": "😐 Neutral",
        "negative": "😔 Negative"
    }

    percentages = []
    for sentiment in ["positive", "neutral", "negative"]:
        count = sentiment_counts.get(sentiment, 0)
        percent = round((count / total_sentiments) * 100, 2)
        percentages.append({
            "Sentiment": sentiment_labels[sentiment],
            "Percentage": percent
        })

    df = pd.DataFrame(percentages)
    
    # Bar chart using Plotly
    fig = px.bar(
        df,
        x="Sentiment",
        y="Percentage",
        text="Percentage",
        color="Sentiment",
        color_discrete_map={
            "😊 Positive": "green",
            "😐 Neutral": "gray",
            "😔 Negative": "red"
        }
    )
    fig.update_layout(title="Sentiment Distribution", title_x=0.5)
    fig.update_traces(textposition='outside')

    return df, fig

# Streamlit UI
st.title("✨ Sentiment Analysis Web App")
st.write("Enter sentences below to analyze their sentiment.")

# Sidebar info with custom profile section
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <style>
        .custom-sidebar {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            width: 650px;
            padding: 10px;
        }
        .profile-container {
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            width: 100%;
        }
        .profile-image {
            width: 200px;
            height: auto;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            margin-right: 15px;
        }
        .profile-details {
            font-size: 14px;
            width: 100%;
        }
        .profile-details h3 {
            margin: 0 0 10px;
            font-size: 18px;
            color: #333;
        }
        .profile-details p {
            margin: 10px 0;
            display: flex;
            align-items: center;
        }
        .profile-details a {
            text-decoration: none;
            color: #1a73e8;
        }
        .profile-details a:hover {
            text-decoration: underline;
        }
        .icon-img {
            width: 18px;
            height: 18px;
            margin-right: 6px;
        }
    </style>

    <div class="custom-sidebar">
        <div class="profile-container">
            <img class="profile-image" src="https://res.cloudinary.com/dwhfxqolu/image/upload/v1744014185/pnhnaejyt3udwalrmnhz.jpg" alt="Profile Image">
            <div class="profile-details">
                <h3>👨‍💻 Developed by:<br> Tahir Abbas Shaikh</h3>
                <p>
                    <img class="icon-img" src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail">
                    <strong>Email:</strong> <a href="mailto:tahirabbasshaikh555@gmail.com">tahirabbasshaikh555@gmail.com</a>
                </p>
                <p>📍 <strong>Location:</strong> Sukkur, Sindh, Pakistan</p>
                <p>
                    <img class="icon-img" src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="GitHub">
                    <strong>GitHub:</strong> <a href="https://github.com/Tahir-Abbas-555" target="_blank">Tahir-Abbas-555</a>
                </p>
                <p>
                    <img class="icon-img" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace">
                    <strong>HuggingFace:</strong> <a href="https://huggingface.co/Tahir5" target="_blank">Tahir5</a>
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# User input
user_input = st.text_area("Enter sentences (one per line):")
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentences = user_input.split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
        sentiments = analyze_sentiment_for_sentences(sentences)
        
        df, fig = calculate_sentiment_percentages(sentiments)

        st.subheader("📊 Sentiment Analysis Results")
        st.dataframe(df, use_container_width=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please enter at least one sentence.")
