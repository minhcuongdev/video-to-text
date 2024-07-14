from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup
import os
import whisper
import datetime
import requests
import uuid

app = FastAPI()

class Transcript:
    def __init__(self,stt,start_time,end_time, text):
        self.stt = stt
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

# Define the upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the Whisper model (you can choose a specific model size like 'base', 'small', 'medium', 'large')
model = whisper.load_model("medium")

@app.post("/transcribe/")
async def transcribe_video( 
    video_url:str,
    language: str = Query(None, description="Language of the video (optional)")
):
    try:
        # Send a request to the TikTok video URL
        response = requests.get(video_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading video: {e}")

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the video tag
    video_tag = soup.find('video')

    print(video_tag)

    if not video_tag:
        raise HTTPException(status_code=400, detail="No video found at the specified URL")

    # Extract the video URL
    video_url = video_tag['src']

    # Download the video
    video_response = requests.get(video_url, stream=True)
    video_response.raise_for_status()

    # Save the video to a temporary file
    file_path = os.path.join(UPLOAD_DIR, uuid.uuid4())

    # Save the video to the specified path
    with open(file_path, 'wb') as file:
        for chunk in video_response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Transcribe the video using Whisper
    transcription = transcribe_video(file_path, language)

    # Delete the video file after transcription
    try:
        os.remove(file_path)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to delete file: {str(e)}"}, status_code=500)

    return JSONResponse(content={"transcription": transcription}, status_code=201)

@app.post("/upload-video/")
async def upload_video(
    file: UploadFile = File(...), 
    language: str = Query(None, description="Language of the video (optional)")
):
    
    # # Check file type
    if not file.filename.endswith(('.mp4', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Transcribe the video using Whisper
    transcription = transcribe_video(file_path, language)

    # Delete the video file after transcription
    try:
        os.remove(file_path)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to delete file: {str(e)}"}, status_code=500)

    return JSONResponse(content={"filename": file.filename,"transcription": transcription}, status_code=201)

def transcribe_video(file_path: str, language: str = 'vi') -> str:
    # Use Whisper to transcribe the video        
    result = model.transcribe(file_path,language = language, fp16=False)
    
    list = []

    for indx, segment in enumerate(result["segments"]):
        list.append(Transcript(indx + 1, start_time= str(datetime.timedelta(seconds = segment['start'])), end_time=str(datetime.timedelta(seconds = segment['end'])),text=segment['text'].strip()).__dict__)
    
    return list

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)