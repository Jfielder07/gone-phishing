"""
EmailAnalysisService
--------------------
Runs the whole analysis for one uploaded file, in order:

    EmailValidator  ->  Email  ->  PhishingDetector  ->  AIExplainer (phishing only)  ->  DatabaseManager

Guarantees:
  * An invalid file stops immediately with a message for the user (EmailValidationError).
  * If Gemini fails, the user still gets the ML classification and score
    (the explanation is marked "unavailable").
  * If the database is missing or fails, the user still gets the full result;
    the problem is only logged.

Usage:
    service = EmailAnalysisService()                 # create once, at server start
    outcome = service.process_email(filename, raw_bytes)
    outcome.result.get_result()                      # what the frontend displays
"""
import logging
import threading
from dataclasses import dataclass, field

from ai_explainer import AIExplainer
from analysis_result import AnalysisResult
from database_manager import DatabaseManager
from email_model import Email
from eml_validator import EmailValidationError, EmailValidator
from phishing_detector import ModelNotAvailableError, PhishingDetector

logger = logging.getLogger("gone_phishing")


class AnalysisUnavailableError(Exception):
    """Raised when emails can't be analyzed at all, i.e. the ML model isn't loaded."""


@dataclass
class AnalysisOutcome:
    """Everything one analysis produced, for main.py to turn into a response."""
    email: Email
    result: AnalysisResult
    findings: dict = field(default_factory=dict)
    saved: bool = False


class EmailAnalysisService:
    def __init__(self, validator=None, detector=None, explainer=None, database=None):
        """
        Each part can be passed in (the tests pass stand-ins); otherwise the real one is created.
        If the trained model can't be loaded, the service still starts, but process_email()
        raises AnalysisUnavailableError until the model is fixed.
        """
        self.validator = validator or EmailValidator()
        self.explainer = explainer or AIExplainer()
        self.database = database or DatabaseManager()
        if detector is not None:
            self.detector = detector
        else:
            try:
                self.detector = PhishingDetector()
            except ModelNotAvailableError as error:
                logger.error("Phishing model not loaded: %s", error)
                self.detector = None

        # The detector and the database connection each keep state between calls, so two
        # analyses running at the same moment take turns using them. Gemini (the slow part)
        # is called outside these locks, so one user's wait doesn't hold up anyone else.
        self._detector_lock = threading.Lock()
        self._database_lock = threading.Lock()

    def process_email(self, filename, content):
        """
        Validate, analyze and save one uploaded file. Returns an AnalysisOutcome.
        Raises EmailValidationError (show its message to the user) or AnalysisUnavailableError.
        """
        self.validator.validate_file(filename, content)  # raises EmailValidationError if invalid
        try:
            email = Email(filename, content)
        except Exception as error:  # the validator accepted it, but extraction still failed
            logger.warning("Could not extract content from %s: %s", filename, error)
            raise EmailValidationError(EmailValidator.BAD_FORMAT_MESSAGE) from error

        result, findings = self.run_analysis(email)
        saved = self.save_analysis(email, result)
        return AnalysisOutcome(email=email, result=result, findings=findings, saved=saved)

    def run_analysis(self, email):
        """
        Classify the email and, only if it looks like phishing, ask for an AI explanation.
        Returns (AnalysisResult, findings dict).
        """
        if self.detector is None:
            raise AnalysisUnavailableError("The phishing detection model isn't available right now.")

        with self._detector_lock:
            outcome = self.detector.analyze_email(email)

        explanation = None
        if outcome["classification"] == AnalysisResult.PHISHING:
            explanation = self._get_explanation(email, outcome)

        # With explanation=None, AnalysisResult marks a phishing result "unavailable",
        # which makes the page show the score plus the "AI service unavailable" notice.
        result = AnalysisResult(outcome["classification"], outcome["safety_score"], explanation=explanation)
        return result, outcome.get("findings", {})

    def save_analysis(self, email, result):
        """
        Save the email and its result. Returns True if saved, False if not.
        Never raises: the user gets their result whether or not saving works.
        """
        if result.classification != AnalysisResult.PHISHING:
            return False  # only phishing emails are stored; safe emails never are (FR-05, NF-03)
        if not self.database.is_configured():
            return False  # no DATABASE_URL yet: nothing to save to (logged once at startup)
        try:
            with self._database_lock:
                email_id = self.database.save_email(email)
                self.database.save_result(result, email_id)
            return True
        except Exception as error:
            logger.error("Could not save the analysis of %s: %s", email.filename, error)
            return False

    # ---- private helpers ----
    def _get_explanation(self, email, outcome):
        """AIExplainer already returns None on failure; this also covers anything unexpected."""
        try:
            return self.explainer.generate_explanation(email, outcome)
        except Exception as error:
            logger.error("AI explainer failed unexpectedly: %s", error)
            return None
