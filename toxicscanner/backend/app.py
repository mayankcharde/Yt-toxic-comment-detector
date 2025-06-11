from flask import Flask, render_template, request
from backend.youtube_scraper import get_comments
from backend.toxicity_detector import predict_toxicity
import pandas as pd
import os

app = Flask(__name__)

comments = []
results = []

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
    global results
    toxic_results = []
    for comment in comments:
        result = predict_toxicity(comment)
        toxic_results.append((
            comment,
            result["weighted_score"],
            result["level"],
            result["scores"]
        ))
    results = toxic_results
    return render_template("index.html", comments=comments, toxic_results=toxic_results)

if __name__ == "__main__":
    app.run(debug=True)
