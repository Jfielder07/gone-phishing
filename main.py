"""
FastAPI backend for Gone Phishing.

The /api/analyze endpoint hands each upload to EmailAnalysisService, which runs
EmailValidator -> Email -> PhishingDetector -> AIExplainer -> DatabaseManager.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Read DATABASE_URL and GEMINI_API_KEY from a .env file when running locally.
# On Render there's no .env file; the values come from the service's environment settings.
load_dotenv()

from database_manager import DatabaseError  # noqa: E402  (imported after load_dotenv on purpose)
from email_analysis_service import AnalysisUnavailableError, EmailAnalysisService  # noqa: E402
from eml_validator import EmailValidationError, EmailValidator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("gone_phishing")
BASE_DIR = Path(__file__).parent
MAX_UPLOAD_BYTES = EmailValidator().max_file_size
service = None  # created at startup, below


@asynccontextmanager
async def lifespan(app):
    # Runs once when the server starts: load the model and check each outside service.
    global service
    service = EmailAnalysisService()

    if service.detector is not None:
        logger.info("Phishing model loaded (trained %s, threshold %s).",
                    service.detector.trained_at, service.detector.threshold)

    if service.explainer.check_availability():
        logger.info("Gemini is reachable (model %s).", service.explainer.model_name)
    else:
        logger.warning("AI explanations unavailable for now: %s", service.explainer.last_error)

    if service.database.is_configured():
        try:
            service.database.connect()
            logger.info("Connected to the database.")
        except DatabaseError as error:
            logger.error("%s", error)  # saving will try again on the next upload
    else:
        logger.warning("DATABASE_URL is not set, so results will not be saved.")

    yield
    # ...and once when it shuts down.
    service.database.close()


app = FastAPI(title="Gone Phishing", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None)):
    filename = file.filename if file else None
    # Read at most one byte over the limit, so a huge upload can't fill up memory;
    # if we got more than the limit, the validator reports it as too large.
    content = await file.read(MAX_UPLOAD_BYTES + 1) if file else b""

    try:
        # The analysis can wait on Gemini for several seconds, so it runs in a worker
        # thread. Other visitors' requests keep being answered meanwhile.
        outcome = await run_in_threadpool(service.process_email, filename, content)
    except EmailValidationError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except AnalysisUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error))

    response = outcome.result.get_result()
    response["saved"] = outcome.saved
    response["findings"] = outcome.findings
    # What the Email class extracted. The page doesn't show this yet; it's here so you
    # can check it (browser dev tools -> Network tab, or http://127.0.0.1:8000/docs).
    email = outcome.email
    response["email"] = {
        "sender": email.get_sender(),
        "subject": email.subject,
        "body_preview": email.get_body()[:200],
        "links": email.links,
    }
    return response
