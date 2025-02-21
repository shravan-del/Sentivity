import os
import praw
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Reddit API credentials from environment variables
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

def get_reddit_sentiment(subreddit="technology", limit=100):
    """Fetch Reddit posts and calculate sentiment scores."""
    posts = [post.title for post in reddit.subreddit(subreddit).hot(limit=limit)]
    sentiment_scores = np.random.rand(len(posts))  # Replace with actual sentiment analysis
    return np.mean(sentiment_scores)

@app.route("/predict", methods=["GET"])
def predict():
    subreddit = request.args.get("subreddit", "technology")
    sentiment_score = get_reddit_sentiment(subreddit)
    return jsonify({"subreddit": subreddit, "sentiment_score": sentiment_score})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Ensure correct port binding for Render
    app.run(host="0.0.0.0", port=port)
