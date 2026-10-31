# Gone Phishing: Email Safety Detector

A web application that checks whether an email is likely to be phishing.
Upload an `.eml` file, click **Analyze**, and get a 0–100 safety score. For emails
flagged as phishing, an AI explanation describes why the email looks unsafe.

> Capstone project, in progress. The ML model and AI explanations are currently placeholders.

## Tech stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI
- **Database:** PostgreSQL (planned)
- **ML model:** scikit-learn (planned)
- **AI explanations:** Google Gemini API (planned)

## Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: serves the page and the `/api/analyze` endpoint |
| `eml_validator.py` | `EmailValidator`: checks the upload is a real `.eml` email |
| `email_model.py` | `Email`: extracts sender, subject, body and links |
| `static/email_upload_interface.js` | `EmailUploadInterface`: handles uploads and displays results |
| `static/index.html`, `static/styles.css` | The web page |
| `test_*.py` | Unit tests |
| `sample_*.eml` | Example emails for testing |

## Running locally

```
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Then open http://127.0.0.1:8000.

## Running the tests

```
python -m unittest test_eml_validator test_email_model -v
```
