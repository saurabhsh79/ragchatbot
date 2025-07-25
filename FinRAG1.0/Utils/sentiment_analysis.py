import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text: str, sentiment_analyzer):
    result = sentiment_analyzer(text[:512])[0]
    label, score = result['label'], result['score']
    sentiment = "Positive" if label == "POSITIVE" else ("Negative" if label == "NEGATIVE" else "Neutral")
    return sentiment, round(score, 3)
