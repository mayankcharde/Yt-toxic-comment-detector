import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_comments(video_id, max_results=100):
    comments = []
    next_page_token = ''
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    while len(comments) < max_results:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "key": API_KEY,
            "textFormat": "plainText",
            "maxResults": min(100, max_results - len(comments)),
            "pageToken": next_page_token
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("Failed to fetch comments:", response.text)
            break

        data = response.json()
        for item in data["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments
