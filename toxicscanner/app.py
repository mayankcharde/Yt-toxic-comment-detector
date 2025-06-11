from flask import Flask, render_template, request
from backend.youtube_scraper import get_comments
from backend.toxicity_detector import predict_toxicity
import pandas as pd
import os
import signal
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

comments = []
results = []

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/fetch-comments", methods=["POST"])
def fetch_comments():
    global comments, results
    url = request.form["video_url"]
    video_id = url.split("v=")[-1]
    comments = get_comments(video_id, max_results=100)
    pd.DataFrame(comments, columns=["Comment"]).to_csv("data/comments.csv", index=False)
    results = []
    return render_template("index.html", comments=comments)

@app.route("/analyze-toxicity", methods=["POST"])
def analyze_toxicity():
    try:
        global results
        toxic_results = []
        stats = {
            "Severely Toxic": 0,
            "Toxic": 0,
            "Non-Toxic": 0
        }
        
        for comment in comments:
            result = predict_toxicity(comment)
            level = result["level"]
            stats[level] += 1
            toxic_results.append((
                comment,
                result["weighted_score"],
                level
            ))
        
        results = toxic_results
        return render_template("index.html", 
                             comments=comments, 
                             toxic_results=toxic_results,
                             stats=stats)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return render_template("index.html", error="An error occurred during analysis")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    try:
        app.run(debug=True, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down gracefully...')
        sys.exit(0)