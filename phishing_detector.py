"""
PhishingDetector
----------------
Runs the trained scikit-learn model on an Email and turns its output into a
classification and a 0-100 safety score (100 = safest).

It also explains, in plain words, what in the email pushed the model toward
"phishing". These findings are what AIExplainer will send to Gemini.

Usage:
    detector = PhishingDetector()            # loads ml/models/phishing_model.joblib once
    outcome = detector.analyze_email(email)  # {"classification": ..., "safety_score": ..., ...}
    detector.get_findings()                  # reasons for the last email analyzed

The model file is created by the training scripts (python -m ml.train_model).
"""
import math
import warnings
from pathlib import Path

import joblib
import sklearn

from analysis_result import AnalysisResult
from ml.features import describe_signals
from ml.text import email_to_text

DEFAULT_MODEL_PATH = Path(__file__).parent / "ml" / "models" / "phishing_model.joblib"


class ModelNotAvailableError(Exception):
    """Raised when the trained model file is missing or can't be loaded."""


class PhishingDetector:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        """Load the trained model. This is slow-ish, so create one detector and reuse it."""
        self.model_path = Path(model_path)
        bundle = self._load_bundle(self.model_path)

        self.model = bundle["pipeline"]        # word features + signals + classifier, in one pipeline
        self.threshold = bundle["threshold"]   # phishing probability at or above this = phishing
        self.safety_score = None               # filled in by analyze_email()
        self.classification = None             # "safe" or "phishing", filled in by analyze_email()

        self.trained_at = bundle.get("trained_at", "unknown")
        self._phishing_probability = None
        self._text = None
        self._email = None

    def analyze_email(self, email):
        """
        Run the whole analysis on one Email object and return a summary dict:
            classification, safety_score, phishing_probability, findings
        The same values are also kept on the detector (self.classification, etc.).
        """
        self._email = email
        self._text = email_to_text(email)
        self._phishing_probability = self.predict_phishing(self._text)
        self.safety_score = self.calculate_score(self._phishing_probability)
        self.classification = (AnalysisResult.PHISHING if self._phishing_probability >= self.threshold
                               else AnalysisResult.SAFE)
        return {
            "classification": self.classification,
            "safety_score": self.safety_score,
            "phishing_probability": round(self._phishing_probability, 4),
            "findings": self.get_findings(),
        }

    def predict_phishing(self, text):
        """Return the model's probability (0.0 to 1.0) that this email text is phishing."""
        return float(self.model.predict_proba([text])[0, 1])

    def calculate_score(self, phishing_probability):
        """Turn a phishing probability into a 0-100 safety score: 0.93 phishing -> 7."""
        probability = min(max(float(phishing_probability), 0.0), 1.0)
        return int(round(100 * (1 - probability)))

    def get_findings(self):
        """
        Plain-language reasons the last analyzed email looks suspicious, plus the
        words that pushed the model hardest toward "phishing". Returns:
            {"signals": [sentences], "suspicious_terms": [words], "links": [urls]}
        """
        if self._text is None:
            raise RuntimeError("Call analyze_email() before get_findings().")
        return {
            "signals": self._describe_signals(describe_signals(self._text)),
            "suspicious_terms": self._top_phishing_terms(),
            "links": list(self._email.links[:10]),
        }

    # ---- private helpers ----
    @staticmethod
    def _load_bundle(path):
        if not path.exists():
            raise ModelNotAvailableError(
                f"No trained model at {path}. Train one with: python -m ml.train_model")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                bundle = joblib.load(path)
            except Exception as error:
                raise ModelNotAvailableError(f"Could not load the model at {path}: {error}") from error
        trained_with = bundle.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            raise ModelNotAvailableError(
                f"The model was trained with scikit-learn {trained_with}, but {sklearn.__version__} is "
                f"installed. Install the same version: pip install scikit-learn=={trained_with}")
        for warning in caught:  # pass on anything unexpected instead of hiding it
            warnings.warn(warning.message, warning.category)
        return bundle

    @staticmethod
    def _describe_signals(s):
        """Turn the raw signal numbers (see ml/features.py) into sentences a person can read."""
        def count(value):  # signals store log(1 + count); undo that
            return int(round(math.expm1(value)))

        reasons = []
        if s["has_ip_link"]:
            reasons.append("Contains a link to a raw IP address instead of a normal website name.")
        if s["has_shortened_link"]:
            reasons.append("Uses a shortened link, which hides where it really goes.")
        if s["has_at_in_link"]:
            reasons.append("A link contains an @ sign, a trick used to disguise the real destination.")
        if count(s["urgency_phrases"]):
            reasons.append(f"Uses urgent or threatening wording ({count(s['urgency_phrases'])} phrases such as "
                           "'verify', 'suspended' or 'act now').")
        if count(s["money_phrases"]):
            reasons.append(f"Mentions accounts, passwords or money ({count(s['money_phrases'])} times).")
        if count(s["click_phrases"]):
            reasons.append("Pushes the reader to click a link ('click here', 'log in here' and similar).")
        if count(s["exclamation_marks"]) >= 3:
            reasons.append(f"Uses many exclamation marks ({count(s['exclamation_marks'])}).")
        if s["capitals_ratio"] >= 0.3:
            reasons.append(f"A large share of the text is in capital letters ({s['capitals_ratio']:.0%}).")
        if count(s["link_count"]) >= 5:
            reasons.append(f"Contains many links ({count(s['link_count'])}).")
        return reasons

    def _top_phishing_terms(self, limit=8):
        """
        The words and phrases in THIS email that pushed the model most toward phishing.
        Only possible for linear models (logistic regression); others return [].
        """
        clf = self.model.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return []
        features = self.model.named_steps["features"]
        vector = features.transform([self._text]).tocsr()
        names = features.get_feature_names_out()
        weights = clf.coef_[0]
        contributions = [(vector.data[i] * weights[j], names[j])
                         for i, j in enumerate(vector.indices) if names[j].startswith("words__")]
        contributions.sort(reverse=True)
        return [name.removeprefix("words__") for value, name in contributions[:limit] if value > 0]
