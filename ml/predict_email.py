"""
Step 4: try the saved model on real .eml files.

Run from the project folder:
    python -m ml.predict_email sample_phishing.eml sample_safe.eml
"""
import sys
from pathlib import Path

from email_model import Email
from ml.data import email_to_text
from ml.features import describe_signals
from ml.pipeline import load_model, safety_score


def main(paths):
    if not paths:
        print("Give one or more .eml files, e.g.  python -m ml.predict_email sample_phishing.eml")
        return
    bundle = load_model()
    pipeline, threshold = bundle["pipeline"], bundle["threshold"]
    print(f"Model trained {bundle['trained_at']}, threshold {threshold}\n")

    for path in paths:
        path = Path(path)
        email = Email(path.name, path.read_bytes())
        text = email_to_text(email)
        probability = pipeline.predict_proba([text])[0, 1]
        verdict = "PHISHING" if probability >= threshold else "safe"
        print(f"{path.name}: {verdict}  (safety score {safety_score(probability)}, "
              f"phishing probability {probability:.3f})")
        print(f"  from: {email.get_sender()}   subject: {email.subject}")
        triggered = {k: round(v, 2) for k, v in describe_signals(text).items() if v and k != "length"}
        print(f"  signals: {triggered or 'none'}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
