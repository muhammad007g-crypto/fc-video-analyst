
import os
import asyncio
import html
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FC Video Analyst")

MODEL_NAME = "gemini-3.8-flash"

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FC Video Analyst</title>
<style>
body {
  font-family: Arial, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  max-width: 760px;
  margin: 30px auto;
  padding: 20px;
}
h1 { color: #34d399; }
input, button {
  width: 100%;
  margin: 18px 0;
  padding: 14px;
  box-sizing: border-box;
}
button {
  background: #34d399;
  border: 0;
  border-radius: 8px;
  font-weight: bold;
  font-size: 16px;
}
.note {
  color: #cbd5e1;
  line-height: 1.6;
}
</style>
</head>
<body>
<h1>FC Video Analyst</h1>
<p>Your personal FC 27 gameplay coach.</p>
<p>Upload your gameplay recording for AI analysis.</p>
<form action="/analyze" method="post"
      enctype="multipart/form-data">
<input type="file" name="video" accept="video/*" required>
<button type="submit">Analyze Gameplay</button>
</form>
<p class="note">
For your first test, use a short 15–30 second gameplay clip.
Long videos may take longer to upload and analyse.
</p>
</body>
</html>
"""


def error_page(message, status_code=500):
    safe_message = html.escape(message)
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
        <title>FC Video Analyst Error</title>
        <style>
        body {{
          font-family: Arial, sans-serif;
          background: #0f172a;
          color: #e2e8f0;
          max-width: 760px;
          margin: 30px auto;
          padding: 20px;
        }}
        h2 {{ color: #f87171; }}
        pre {{
          white-space: pre-wrap;
          overflow-wrap: anywhere;
          background: #1e293b;
          padding: 15px;
          border-radius: 8px;
        }}
        a {{ color: #34d399; }}
        </style>
        </head>
        <body>
        <h2>Analysis failed</h2>
        <p>The video could not be analysed.</p>
        <p>Error details:</p>
        <pre>{safe_message}</pre>
        <p><a href="/">Return to FC Video Analyst</a></p>
        </body>
        </html>
        """,
        status_code=status_code,
    )


@app.get("/", response_class=HTMLResponse)
async def home():
    return PAGE


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(video: UploadFile = File(...)):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return error_page(
            "GEMINI_API_KEY is missing from Render Environment.",
            500,
        )

    filename = video.filename or "gameplay.mp4"
    suffix = Path(filename).suffix.lower()

    allowed_formats = {
        ".mp4", ".mov", ".webm", ".mpeg", ".mpg"
    }

    if suffix not in allowed_formats:
        await video.close()
        return error_page(
            "Unsupported video format. Use MP4, MOV, WebM or MPEG.",
            400,
        )

    temp_path = None
    gemini_file = None
    client = None

    prompt = """
You are an expert EA SPORTS FC 27 gameplay coach.

Analyse the uploaded gameplay video carefully. Base your conclusions
on events actually visible in the footage. Never invent events,
timestamps, player inputs or match details.

Write a practical coaching report using these sections:

1. MATCH OVERVIEW
Describe the visible match situation and general play patterns.

2. PASSING AND DECISION-MAKING
Identify rushed passes, missed passing options and good decisions.

3. DEFENDING
Assess defensive positioning, pressure, jockeying and gaps when
the footage provides enough evidence.

4. PLAYER SWITCHING AND POSITIONING
Discuss observable switching and positioning mistakes.
Do not claim to know controller inputs that are not visible.

5. ATTACKING
Assess build-up, movement, chance creation and final decisions.

6. TOP THREE MISTAKES
Explain each mistake, give an approximate video timestamp if
possible, and recommend a better action.

7. PERSONAL TRAINING PLAN
Give three specific drills for the next practice session.

8. WHAT THE PLAYER DID WELL
Identify strengths supported by the footage.

Use clear language and actionable advice. Be honest about uncertainty.
If the clip does not show enough evidence for a conclusion, say so.
"""

    try:
        client = genai.Client(api_key=api_key)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:
            temp_path = temp.name

            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                temp.write(chunk)

        if os.path.getsize(temp_path) == 0:
            return error_page("The uploaded video is empty.", 400)

        logger.info("Uploading gameplay video to Gemini.")

        gemini_file = await run_in_threadpool(
            client.files.upload,
            file=temp_path,
        )

        # Wait for Gemini to finish processing the video.
        for attempt in range(180):
            current_state = getattr(
                gemini_file, "state", None
            )
            state_name = getattr(
                current_state, "name", str(current_state or "")
            ).upper()

            if state_name.endswith("ACTIVE"):
                break

            if state_name.endswith("FAILED"):
                raise RuntimeError(
                    "Gemini reported that video processing failed."
                )

            if attempt == 179:
                raise RuntimeError(
                    "Video processing exceeded six minutes. "
                    "Try a shorter clip."
                )

            await asyncio.sleep(2)

            gemini_file = await run_in_threadpool(
                client.files.get,
                name=gemini_file.name,
            )

        logger.info("Video is ready. Requesting coaching report.")

        response = await run_in_threadpool(
            client.models.generate_content,
            model=MODEL_NAME,
            contents=[gemini_file, prompt],
        )

        report = response.text

        if not report:
            raise RuntimeError(
                "Gemini returned no report text. Please try again."
            )

        safe_report = html.escape(report)

        return HTMLResponse(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta name="viewport"
                  content="width=device-width, initial-scale=1">
            <title>FC Coaching Report</title>
            <style>
            body {{
              font-family: Arial, sans-serif;
              background: #0f172a;
              color: #e2e8f0;
              max-width: 760px;
              margin: 30px auto;
              padding: 20px;
            }}
            h1 {{ color: #34d399; }}
            pre {{
              white-space: pre-wrap;
              overflow-wrap: anywhere;
              line-height: 1.7;
              background: #1e293b;
              padding: 18px;
              border-radius: 10px;
              font-family: Arial, sans-serif;
            }}
            a {{ color: #34d399; }}
            </style>
            </head>
            <body>
            <h1>Your FC 27 Coaching Report</h1>
            <pre>{safe_report}</pre>
            <p><a href="/">Analyse another video</a></p>
            </body>
            </html>
            """
        )

    except Exception as exc:
        logger.exception("Gemini gameplay analysis failed")
        return error_page(str(exc), 500)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not remove temporary video file.")

        if client is not None and gemini_file is not None:
            try:
                await run_in_threadpool(
                    client.files.delete,
                    name=gemini_file.name,
                )
            except Exception:
                logger.warning(
                    "Could not delete temporary Gemini file."
                )

        await video.close()
      
