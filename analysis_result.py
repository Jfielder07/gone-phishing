"""
AnalysisResult
--------------
Data object holding the outcome of analyzing one email: the ML model's
classification and safety score, plus the LLM's explanation if there is one.

explanation_status records what happened with the explanation:
    "not_needed"   the email is safe, so no explanation was requested
    "available"    the email is phishing and the LLM explained why
    "unavailable"  the email is phishing but the LLM failed or was unreachable

Usage:
    result = AnalysisResult("phishing", 12, explanation=None)
    result.explanation_status   # 'unavailable'
    result.get_result()         # dict sent to the frontend as JSON
"""


class AnalysisResult:
    SAFE = "safe"
    PHISHING = "phishing"
    VALID_CLASSIFICATIONS = (SAFE, PHISHING)

    STATUS_NOT_NEEDED = "not_needed"
    STATUS_AVAILABLE = "available"
    STATUS_UNAVAILABLE = "unavailable"
    VALID_STATUSES = (STATUS_NOT_NEEDED, STATUS_AVAILABLE, STATUS_UNAVAILABLE)

    def __init__(self, classification, safety_score, explanation=None, explanation_status=None, result_id=None):
        """
        classification:     "safe" or "phishing", from PhishingDetector
        safety_score:       0-100, where 100 is safest
        explanation:        the LLM's explanation text, or None
        explanation_status: leave as None to work it out automatically (see the top of this file)
        result_id:          the database ID; None until DatabaseManager saves the result
        """
        if classification not in self.VALID_CLASSIFICATIONS:
            raise ValueError(f"classification must be 'safe' or 'phishing', got {classification!r}")

        score = float(safety_score)  # also raises ValueError for things like "abc"
        if not 0 <= score <= 100:
            raise ValueError(f"safety_score must be between 0 and 100, got {safety_score!r}")

        # Treat blank text the same as no explanation at all
        if explanation is not None:
            explanation = str(explanation).strip() or None

        if explanation_status is None:
            explanation_status = self._work_out_status(classification, explanation)
        elif explanation_status not in self.VALID_STATUSES:
            raise ValueError(f"explanation_status must be one of {self.VALID_STATUSES}, got {explanation_status!r}")

        self.result_id = result_id
        self.classification = classification
        self.safety_score = round(score)  # whole numbers are clearer to users than 87.36
        self.explanation = explanation if explanation_status == self.STATUS_AVAILABLE else None
        self.explanation_status = explanation_status

    def get_result(self):
        """Return everything as a dictionary, ready to send to the frontend as JSON."""
        return {
            "result_id": self.result_id,
            "classification": self.classification,
            "safety_score": self.safety_score,
            "explanation": self.explanation,
            "explanation_status": self.explanation_status,
            # The frontend checks this true/false field to decide what to display
            "explanation_available": self.explanation_status == self.STATUS_AVAILABLE,
        }

    def get_score(self):
        """Return the safety score, a whole number from 0 (dangerous) to 100 (safe)."""
        return self.safety_score

    def get_explanation(self):
        """Return the LLM's explanation, or None if there isn't one."""
        return self.explanation

    def is_phishing(self):
        """Convenience check, e.g. for deciding whether to ask the LLM for an explanation."""
        return self.classification == self.PHISHING

    # ---- private helpers ----
    @classmethod
    def _work_out_status(cls, classification, explanation):
        if classification == cls.SAFE:
            return cls.STATUS_NOT_NEEDED
        return cls.STATUS_AVAILABLE if explanation else cls.STATUS_UNAVAILABLE

    def __repr__(self):
        return (f"AnalysisResult(id={self.result_id}, classification={self.classification!r}, "
                f"score={self.safety_score}, explanation_status={self.explanation_status!r})")
