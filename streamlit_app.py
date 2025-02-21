import os
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import streamlit as st
import joblib

# API URL for backend
API_URL = "https://sentivity.onrender.com/predict?subreddit="  # Update with Render API URL

st.title("📊 7-Day Sentiment Forecast")

# List of subreddits to analyze
subreddits = ["centrist", "wayofthebern", "libertarian", "conservatives", "progun"]

# Function to fetch sentiment from API
def get_sentiment(subreddit):
    try:
        response = requests.get(API_URL + subreddit, timeout=10)
        data = response.json()
        return data.get("sentiment_score", 0)
    except:
        return 0  # Default to 0 if API call fails

if st.button("Run Sentiment Analysis"):
    with st.spinner("Fetching sentiment scores..."):
        scores = {sub: get_sentiment(sub) for sub in subreddits}
    
    st.success("Analysis complete!")
    
    # Convert data into DataFrame
    df = pd.DataFrame(scores.items(), columns=["Subreddit", "Sentiment Score"])
    st.write("### Sentiment Scores")
    st.dataframe(df)
    
    # Plot the data
    fig, ax = plt.subplots()
    ax.bar(df["Subreddit"], df["Sentiment Score"], color="blue")
    ax.set_title("Sentiment Analysis by Subreddit")
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Subreddit")
    st.pyplot(fig)
    
    # Provide Download Button for CSV Export
    st.download_button(
        label="📥 Download Sentiment Data as CSV",
        data=df.to_csv(index=False),
        file_name="sentiment_scores.csv",
        mime="text/csv",
    )
