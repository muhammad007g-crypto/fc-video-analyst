
import os
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI(title="FC Video Analyst")

MAX_VIDEO_SIZE = 2 * 1024 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>FC Video Analyst</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #101827;
                color: white;
                max-width: 650px;
                margin: 40px auto;
                padding: 20px;
            }
            h1 { color: #48e09b; }
            input, button {
                display: block;
                width: 100%;
                box-sizing: border-box;
                margin-top: 18px;
                padding: 14px;
                border-radius: 8px;
            }
            button {
                background: #48e09b;
                border: none;
                font-weight: bold;
            }
            p { line-height: 1.6; color: #d1d5db; }
        </style>
    </head>
    <body>
        <h1>FC Video Analyst</h1>
        <p>Your personal FC 27 gameplay coach.</p>
        <p>Upload a gameplay recording to begin.</p>
        <form action="/analyze" method="post"
              enctype="multipart/form-data">
            <input type="file" name="video"
                   accept="video/*" required>
            <button type="submit">Analyze Gameplay</button>
        </form>
        <p>Maximum intended file size: 2 GB.</p>
        <p>AI coaching analysis is coming next.</p>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "ok", "app": "FC Video Analyst"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    suffix = Path(video.filename or "").suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload an MP4, MOV, MKV, or WEBM video."
        )

    temp_path = None

    try:
        total_size = 0

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp:
            temp_path = temp.name

            while True:
                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_VIDEO_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 2 GB limit."
                    )

                temp.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty."
            )

        capture = cv2.VideoCapture(temp_path)

        try:
            if not capture.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail="Could not read this video."
                )

            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

            duration = frames / fps if fps > 0 else 0

            return {
                "status": "uploaded",
                "filename": video.filename,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "duration_seconds": round(duration, 2),
                "message": (
                    "Video received. Gameplay AI analysis "
                    "will be added in a later step."
                )
            }
        finally:
            capture.release()

    finally:
        await video.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
          
