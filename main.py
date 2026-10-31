"""
Temporary FastAPI backend for testing EmailUploadInterface.

The /api/analyze endpoint returns a FAKE result for now. Later it will call
EmailAnalysisService (validation -> PhishingDetector -> AIExplainer -> DatabaseManager).

Run:  uvicorn main:app --reload
Open: http://127.0.0.1:8000
"""
import random
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from email_model import Email
from eml_validator import EmailValidationError, EmailValidator

BASE_DIR = Path(__file__).parent
app = FastAPI(title="Gone Phishing")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
validator = EmailValidator()


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None)):
    filename = file.filename if file else None
    # Read at most one byte over the limit, so a huge upload can't fill up memory;
    # if we got more than the limit, the validator reports it as too large.
    content = await file.read(validator.max_file_size + 1) if file else b""

    try:
        validator.validate_file(filename, content)
    except EmailValidationError as error:
        raise HTTPException(status_code=400, detail=str(error))

    # Turn the validated file into an Email object (PhishingDetector will analyze this later)
    email = Email(filename, content)

    # ---- Placeholder result (replace with PhishingDetector + AIExplainer) ----
    score = random.randint(0, 100)
    is_safe = score >= 50
    llm_available = random.choice([True, False])  # simulates the Gemini service being up/down
    return {
        "classification": "safe" if is_safe else "phishing",
        "safety_score": score,
        "explanation": None if is_safe or not llm_available
        else "Placeholder explanation: the sender domain doesn't match the display name and the email urges immediate action.",
        "explanation_available": (not is_safe) and llm_available,
        # What the Email class extracted. The page doesn't show this yet; it's here so you
        # can check it (browser dev tools -> Network tab, or http://127.0.0.1:8000/docs).
        "email": {
            "sender": email.get_sender(),
            "subject": email.subject,
            "body_preview": email.get_body()[:200],
            "links": email.links,
        },
    }
