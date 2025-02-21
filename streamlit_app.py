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

# Function to generate forecast plot
def generate_forecast_plot(pred, output_path='forecast_plot.png'):
    today = datetime.date.today()
    days = [today + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]
    
    xnew = np.linspace(0, 6, 300)
    spline = make_interp_spline(np.arange(7), pred, k=3)
    pred_smooth = spline(xnew)
    
    plt.figure(figsize=(12, 7))
    plt.fill_between(xnew, pred_smooth, color='#aec7e8', alpha=0.4)
    plt.plot(xnew, pred_smooth, color='#1f77b4', lw=3, label='Forecast')
    plt.scatter(np.arange(7), pred, color='#1f77b4', s=100, zorder=5)
    plt.title("7-Day Sentiment Forecast", fontsize=22, fontweight='bold', pad=20)
    plt.xlabel("Day", fontsize=16)
    plt.ylabel("Negative Sentiment", fontsize=16)
    plt.xticks(np.arange(7), days_str, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

# Run sentiment analysis on button click
if st.button("Run Sentiment Analysis"):
    with st.spinner("Fetching sentiment scores..."):
        scores = {sub: get_sentiment(sub) for sub in subreddits}
    
    st.success("Analysis complete!")
    
    # Convert data into DataFrame
    df = pd.DataFrame(scores.items(), columns=["Subreddit", "Sentiment Score"])
    st.write("### Sentiment Scores")
    st.dataframe(df)
    
    # Generate and display the forecast plot
    pred = np.random.rand(7)  # Replace with actual sentiment prediction logic
    plot_path = generate_forecast_plot(pred)
    st.image(plot_path, caption="7-Day Sentiment Forecast", use_column_width=True)

    # Provide Download Button for CSV Export
    st.download_button(
        label="📥 Download Sentiment Data as CSV",
        data=df.to_csv(index=False),
        file_name="sentiment_scores.csv",
        mime="text/csv",
    )
