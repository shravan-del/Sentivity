import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import praw

app = Flask(__name__)
CORS(app)

# File paths
DATA_FILE = "sentiment_data.json"
GRAPH_FILE = "graph.png"

# Load Reddit API credentials
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

if not all([client_id, client_secret, user_agent]):
    raise ValueError("Missing Reddit API credentials. Ensure environment variables are set.")

# Set up Reddit API
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    check_for_async=False
)

# Function to fetch sentiment scores
def get_reddit_sentiment(subreddit="technology", limit=100):
    posts = [post.title for post in reddit.subreddit(subreddit).hot(limit=limit)]
    sentiment_scores = np.random.rand(len(posts))  # Replace with actual sentiment analysis
    return np.mean(sentiment_scores)

# Function to generate the sentiment graph
def generate_forecast_plot(sentiments):
    today = datetime.date.today()
    days = [today - datetime.timedelta(days=i) for i in range(len(sentiments))]
    days_str = [day.strftime('%a %m/%d') for day in days][::-1]  # Reverse for chronological order

    # Smooth the curve using spline interpolation
    xnew = np.linspace(0, len(sentiments) - 1, 300)
    spline = make_interp_spline(np.arange(len(sentiments)), sentiments, k=3)
    pred_smooth = spline(xnew)

    plt.figure(figsize=(12, 7))
    plt.fill_between(xnew, pred_smooth, color='#aec7e8', alpha=0.4)
    plt.plot(xnew, pred_smooth, color='#1f77b4', lw=3, label="Forecast")
    plt.scatter(np.arange(len(sentiments)), sentiments, color='#1f77b4', s=100, zorder=5)
    plt.title("7-Day Sentiment Forecast", fontsize=22, fontweight='bold', pad=20)
    plt.xlabel("Day", fontsize=16)
    plt.ylabel("Negative Sentiment", fontsize=16)
    plt.xticks(np.arange(len(sentiments)), days_str, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(GRAPH_FILE)
    plt.close()

# Function to update sentiment data and generate graph
def update_sentiment_data():
    today = str(datetime.date.today())

    # Fetch sentiment scores for subreddits
    subreddits = ["technology", "science", "politics"]
    sentiment_data = {sub: get_reddit_sentiment(sub) for sub in subreddits}

    # Save data
    data = {"date": today, "sentiments": sentiment_data}
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

    # Generate graph
    generate_forecast_plot(list(sentiment_data.values()))

    return data

# Route to get the latest sentiment data
@app.route("/sentiment", methods=["GET"])
def get_sentiment():
    if not os.path.exists(DATA_FILE):
        update_sentiment_data()  # Generate the first data file if missing

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    return jsonify(data)

# Route to serve the sentiment graph
@app.route("/graph.png", methods=["GET"])
def get_graph():
    if not os.path.exists(GRAPH_FILE):
        update_sentiment_data()  # Ensure graph exists
    return send_file(GRAPH_FILE, mimetype="image/png")

# Route to manually update data (For Cron Job)
@app.route("/update", methods=["GET"])
def update():
    data = update_sentiment_data()
    return jsonify({"message": "Sentiment data updated!", "data": data})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
