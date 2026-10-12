
import os
import asyncio
import html
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from google import genai

app = FastAPI(title="FC Video Analyst")

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FC Video Analyst</title>
<style>
body {
  font-family: Arial, sans-serif;
  background: #0f172a; color: #e2e8f0;
  max-width: 760px; margin: 50px auto; padding: 20px;
}
h1 { color: #34d399; }
input, button { width: 100%; margin: 18px 0; padding: 14px;
  box-sizing: border-box; }
button { background: #34d399; border: 0; border-radius: 8px;
  font-weight: bold; font-size: 16px; }
.result { white-space: pre-wrap; line-height: 1.7;
  background: #1e293b; padding: 18px; border-radius: 10px; }
a { color: #34d399; }
</style>
</head>
<body>
<h1>FC Video Analyst</h1>
<p>Your personal FC 27 gameplay coach.</p>
<p>Upload your gameplay recording for AI analysis.</p>
<form action="/analyze" method="post" enctype="multipart/form-data">
<input type="file" name="video" accept="video/*" required>
<button type="submit">Analyze Gameplay</button>
</form>
<p>Analysis may take several minutes for long videos.</p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return PAGE


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(video: UploadFile = File(...)):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return HTMLResponse(
            "<h2>Configuration error</h2>"
            "<p>Gemini API key is missing from Render.</p>",
            status_code=500
        )

    suffix = Path(video.filename or "gameplay.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mpeg", ".mpg"}:
        return HTMLResponse(
            "<h2>Unsupported video format</h2>"
            "<p>Please upload an MP4, MOV, WebM or MPEG video.</p>",
            status_code=400
        )

    temp_path = None
    gemini_file = None
    client = genai.Client(api_key=api_key)

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp:
            temp_path = temp.name
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                temp.write(chunk)

        prompt = """
You are an expert EA SPORTS FC 27 gameplay coach.

Analyse the uploaded match video carefully. Base your conclusions
on what is actually visible. Do not invent events or timestamps.

Give the player a practical coaching report with these sections:

1. MATCH OVERVIEW
Describe the visible match situation and general play patterns.

2. PASSING AND DECISION-MAKING
Identify observable rushed passes, missed options, or good decisions.

3. DEFENDING
Analyse defensive positioning, pressure, jockeying, and gaps where
the footage provides enough evidence.

4. PLAYER SWITCHING AND POSITIONING
Point out observable switching or positioning mistakes, if any.

5. ATTACKING
Assess movement, build-up, chance creation, and final decisions.

6. TOP THREE MISTAKES
For each mistake, give an approximate video timestamp if possible,
explain why it matters, and suggest a better action.

7. PERSONAL TRAINING PLAN
Give three specific drills or practice tasks for the next session.

8. WHAT THE PLAYER DID WELL
Identify strengths supported by the footage.

Be honest about uncertainty. If an event or control input cannot
be determined from the video, say so. Do not claim to know the
controller inputs when they are not visible.
Use clear language and actionable advice.
"""

        gemini_file = await run_in_threadpool(
            client.files.upload, file=temp_path
        )

        # Wait for Gemini to finish processing the video.
        for _ in range(90):
            state = str(
                getattr(gemini_file, "state", "")
            ).upper()

            if state.endswith("ACTIVE"):
                break
            if state.endswith("FAILED"):
                raise RuntimeError("Gemini could not process the video.")

            await asyncio.sleep(2)
            gemini_file = await run_in_threadpool(
                client.files.get, name=gemini_file.name
            )
        else:
            raise RuntimeError(
                "Video processing took too long. Try a shorter clip."
            )

        response = await run_in_threadpool(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=[gemini_file, prompt],
        )

        report = response.text or "No analysis text was returned."
        safe_report = html.escape(report)

        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html><head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>FC Coaching Report</title>
        <style>
        body {{ font-family: Arial; background:#0f172a; color:#e2e8f0;
          max-width:760px; margin:30px auto; padding:20px; }}
        h1 {{ color:#34d399; }}
        pre {{ white-space:pre-wrap; overflow-wrap:anywhere;
          line-height:1.7; background:#1e293b; padding:18px;
          border-radius:10px; font-family:Arial; }}
        a {{ color:#34d399; }}
        </style></head><body>
        <h1>Your FC 27 Coaching Report</h1>
        <pre>{safe_report}</pre>
        <p><a href="/">Analyse another video</a></p>
        </body></html>
        """)

    except Exception:
        return HTMLResponse(
            "<h2>Analysis failed</h2>"
            "<p>Check your Render logs for the error. "
            "Verify your Gemini API access and try a shorter video.</p>"
            '<p><a href="/">Return to FC Video Analyst</a></p>',
            status_code=500
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if gemini_file is not None:
            try:
                await run_in_threadpool(
                    client.files.delete, name=gemini_file.name
                )
            except Exception:
                pass
        await video.close()
        
