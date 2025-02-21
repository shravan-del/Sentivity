import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import praw
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import streamlit as st
import time
from flask import Flask, request, jsonify
import threading
import os
import json


# Securely get API keys from Streamlit secrets
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/xlm-twitter-politics-sentiment"
API_TOKEN = st.secrets["HF_API_TOKEN"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Reddit API authentication
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
    user_agent=st.secrets["REDDIT_USER_AGENT"],
    check_for_async=False
)

# List of subreddits to analyze
subreddits = [
    "centrist",
    "wayofthebern",
    "libertarian",
    "conservatives",
    "progun"
]

# Analysis period: past 14 days (UTC dates)
end_date = datetime.datetime.utcnow().date()  # today in UTC
start_date = end_date - datetime.timedelta(days=14)  # 14 days ago
start_timestamp = int(datetime.datetime.combine(start_date, datetime.time.min).timestamp())
end_timestamp = int(datetime.datetime.combine(end_date, datetime.time.max).timestamp())

# --- Functions ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def query(text_list, batch_size=5, min_request_interval=10):
    """Batch process multiple posts at once while ensuring a minimum request interval (rate limit)."""
    global last_request_time
    results = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        payload = {"inputs": batch}

        # Ensure at least min_request_interval seconds have passed since last request
        time_since_last_request = time.time() - last_request_time
        if time_since_last_request < min_request_interval:
            sleep_time = min_request_interval - time_since_last_request
            print(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raise an exception for HTTP errors
            results.extend(response.json())
            last_request_time = time.time()  # Update last request time
        except requests.exceptions.RequestException as e:
            print(f"API request failed for batch {i//batch_size + 1}: {e}")
            # Skip this batch and continue with the next one
            results.extend([{"error": "Skipped due to API failure"}] * len(batch))

    return results

def get_negative_score(text):
    """
    Given a text string, returns the sentiment score for the 'Negative' label.
    If something goes wrong or the label is not found, returns 0.0.
    """
    try:
        post = query([text])
        for item in post:
            itemized = str(item)
            if len(item) == 0:
                negative_scores = 0.0
            else:
                score_index = itemized.find("'Negative'") + 23
                negative_scores = float(itemized[score_index-1:score_index+2])
    except Exception as e:
        #print(f"Error processing text: {text}\nError: {e}")
        return 0.0
    return negative_scores

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_daily_posts_sentiments(subreddit_name, day_start_ts, day_end_ts, max_posts=5, limit=10):
    """
    For a given subreddit and a specific day (defined by day_start_ts and day_end_ts),
    fetch up to max_posts posts from the subreddit's 'new' feed (using a higher overall limit to allow filtering).
    Returns a list of tuples: (post_date, negative_sentiment_score)
    """
    sentiments = []
    subreddit_obj = reddit.subreddit(subreddit_name)
    
    # Collect posts within the specified time range
    query = f"timestamp:{day_start_ts}..{day_end_ts}"
    submissions = list(subreddit_obj.search(query, sort="new", limit=limit))

    # Extract post titles for batch API query
    post_dates = [datetime.datetime.utcfromtimestamp(s.created_utc).date() for s in submissions]
    titles = [s.title for s in submissions]

    # Process each title to get the negative sentiment score
    for post_date, title in zip(post_dates, titles):
        negative_score = get_negative_score(title)
        sentiments.append((post_date, negative_score))
        if len(sentiments) >= max_posts:
            break  # Stop once we have enough posts

    return sentiments


# --- Streamlit App ---
st.title("📊 7-Day Sentiment Forecast")

app = Flask(__name__)

# Load pre-trained model
model = joblib.load("multioutput_regressor_model.pkl")

# Set up Reddit API (Replace placeholders with your credentials)
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_USER_AGENT"
)

def get_reddit_sentiment(subreddit="technology", limit=100):
    """Fetches Reddit posts and calculates sentiment scores."""
    posts = [post.title for post in reddit.subreddit(subreddit).hot(limit=limit)]
    sentiment_scores = np.random.rand(len(posts))  # Replace with actual sentiment analysis
    return np.mean(sentiment_scores)  # Example: Return average sentiment

@app.route("/predict", methods=["GET"])
def predict():
    subreddit = request.args.get("subreddit", "technology")
    sentiment_score = get_reddit_sentiment(subreddit)
    return jsonify({"subreddit": subreddit, "sentiment_score": sentiment_score})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


if st.button("Run Sentiment Analysis"):
    with st.spinner("Analyzing sentiment..."):
        all_sentiments = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for day_offset in range(14):
            current_day = start_date + datetime.timedelta(days=day_offset)
            day_start = datetime.datetime.combine(current_day, datetime.time.min)
            day_end = datetime.datetime.combine(current_day, datetime.time.max)
            day_start_ts = int(day_start.timestamp())
            day_end_ts = int(day_end.timestamp())

            for sub in subreddits:
                status_text.text(f"Processing {sub} for {current_day}...")
                daily_sentiments = fetch_daily_posts_sentiments(sub, day_start_ts, day_end_ts, max_posts=5, limit=5)
                all_sentiments.extend(daily_sentiments)

            progress_bar.progress((day_offset + 1) / 14)
            time.sleep(1)  # Simulate delay for demonstration

        status_text.text("Analysis complete!")
        st.success("Analysis complete!")

    # Grouping and Averaging by Day
    daily_scores = {start_date + datetime.timedelta(days=i): [] for i in range(14)}
    for (post_date, score) in all_sentiments:
        if post_date in daily_scores:
            daily_scores[post_date].append(score)

    avg_daily_scores = []
    days = []
    for i in range(14):
        day = start_date + datetime.timedelta(days=i)
        days.append(day)
        if daily_scores[day]:
            avg = np.mean(daily_scores[day])
        else:
            avg = 0.0  # No posts for that day: default to 0.
        avg_daily_scores.append(avg)

    nonzero_values = [v for v in avg_daily_scores if v > 0]
    mean_nonzero = np.mean(nonzero_values) if nonzero_values else 0.0
    avg_daily_scores = [mean_nonzero if v == 0 else v for v in avg_daily_scores]

    # Load pre-trained model
    MODEL_FILE = "multioutput_regressor_model.pkl"
    model = joblib.load(MODEL_FILE)

    # Function to ensure correct input shape before prediction
    def convert_to_model_input(input_array):
        arr = np.array(input_array)
        if arr.ndim != 1 or arr.shape[0] != 14:
            raise ValueError("Input array must be one-dimensional with exactly 14 elements.")
        return arr.reshape(1, -1)

    # Generate predictions
    x = convert_to_model_input(avg_daily_scores)
    pred = model.predict(x)[0]

    # Generate Dates for Forecast
    today = datetime.date.today()
    days = [today + datetime.timedelta(days=i) for i in range(7)]
    days_str = [day.strftime('%a %m/%d') for day in days]

    # Smooth the curve using spline interpolation
    x_smooth = np.linspace(0, 6, 100)  # Faster plotting
    spline = make_interp_spline(np.arange(7), pred, k=3)
    pred_smooth = spline(x_smooth)

    # Create Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(x_smooth, pred_smooth, color='#aec7e8', alpha=0.4)
    ax.plot(x_smooth, pred_smooth, color='#1f77b4', lw=3, label='Forecast')
    ax.scatter(np.arange(7), pred, color='#1f77b4', s=100, zorder=5)

    ax.set_title("7-Day Sentiment Forecast", fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel("Day", fontsize=16)
    ax.set_ylabel("Negative Sentiment", fontsize=16)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(days_str, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)  # Set fontsize of y-axis tick labels
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=14)

    # Display the graph in Streamlit
    st.pyplot(fig)

    
    # Create Forecast DataFrame
    df_forecast = pd.DataFrame({"Date": days_str, "Predicted Sentiment": pred})

    # Display the table
    st.write("### 📋 Forecast Data")
    st.dataframe(df_forecast)

    # Provide Download Button for CSV Export
    st.download_button(
        label="📥 Download Forecast Data as CSV",
        data=df_forecast.to_csv(index=False),
        file_name="sentiment_forecast.csv",
        mime="text/csv",
    )
    

