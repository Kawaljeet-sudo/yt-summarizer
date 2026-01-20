from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from schemas import VideoRequest
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import requests
import os

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],  # allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],
)

ytt_api = YouTubeTranscriptApi()

def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)

    if "youtube.com" in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")
    
    return None

def get_transcript(video_id: str) -> str | None:
    """
    Fetch the transcript of a YouTube video using ytt_api.
    Returns the full transcript as a single string, or None if unavailable.
    """
    try:
        transcript = ytt_api.fetch(video_id)

        if not transcript:
            return None

        transcript_list = transcript.to_raw_data()
        full_text = " ".join(entry['text'] for entry in transcript_list)
        return full_text

    except Exception as e:
        print(f"[get_transcript] Error fetching transcript for video_id={video_id}: {e}")
        return None

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set in environment variables")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

def summarize_text(text: str) -> str:
    prompt = (
        "Summarize the following YouTube transcript into EXACTLY 10 concise bullet points. Use bold titles where helpful. No intro or outro text. Add a conclusion at the end too."
        f"{text}"
    )

    payload = {
        "model": "google/gemma-3-27b-it:featherless-ai",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("[summarize_text] Error:", e)
        return "Failed to generate summary."

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/summarize")
def summarize_video(data: VideoRequest):
    video_id = extract_video_id(data.youtube_url)

    if not video_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid YouTube URL"
        )
    
    transcript = get_transcript(video_id)

    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not available for this video"
        )
    
    # summarize the transcript
    summary = summarize_text(transcript)


    return {
        "youtube_url": data.youtube_url,
        "video_id": video_id,
        "summary": summary
    }
