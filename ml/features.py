"""
Hand-made phishing signals, used alongside the word features.

The word features (TF-IDF) learn which words appear in phishing emails. These
signals add things words alone miss: links pointing at raw IP addresses,
shortened links, how much of the email is in CAPITALS, and so on.

The saved model refers to EmailSignals by name, so this file must stay at
ml/features.py for the model to load later.
"""
import math
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)
UPPER_RE = re.compile(r"[A-Z]")
LETTER_RE = re.compile(r"[A-Za-z]")
HTML_RE = re.compile(r"<\s*(?:html|a\s+href|table|div|font)\b", re.IGNORECASE)

SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co/", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly")
URGENCY_PHRASES = (
    "urgent", "immediately", "suspended", "verify", "within 24 hours", "within 48 hours", "act now",
    "limited time", "expire", "locked", "unusual activity", "confirm your", "update your",
    "security alert", "final notice", "action required",
)
MONEY_PHRASES = (
    "bank", "account", "password", "credit card", "social security", "wire transfer", "payment",
    "invoice", "refund", "lottery", "prize", "winner", "inheritance", "bitcoin", "gift card", "million",
)
CLICK_PHRASES = ("click here", "click below", "click the link", "follow the link", "log in here", "login here")

FEATURE_NAMES = [
    "link_count", "has_ip_link", "has_shortened_link", "has_at_in_link", "urgency_phrases",
    "money_phrases", "click_phrases", "exclamation_marks", "capitals_ratio", "has_html", "length",
]


class EmailSignals(BaseEstimator, TransformerMixin):
    """scikit-learn step that turns each email's text into the numbers in FEATURE_NAMES."""

    def fit(self, X, y=None):
        return self  # nothing to learn: the signals are fixed rules

    def transform(self, X):
        return np.array([signals_for(str(text)) for text in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES, dtype=object)


def signals_for(text):
    """Return the list of signal values for one email's text."""
    lower = text.lower()
    urls = URL_RE.findall(text)
    letters = len(LETTER_RE.findall(text))
    return [
        math.log1p(len(urls)),                                  # counts are log-scaled so 200 links
        float(bool(IP_URL_RE.search(text))),                    # doesn't swamp everything else
        float(any(s in lower for s in SHORTENERS)),
        float(any("@" in u for u in urls)),
        math.log1p(sum(lower.count(p) for p in URGENCY_PHRASES)),
        math.log1p(sum(lower.count(p) for p in MONEY_PHRASES)),
        math.log1p(sum(lower.count(p) for p in CLICK_PHRASES)),
        math.log1p(text.count("!")),
        len(UPPER_RE.findall(text)) / letters if letters else 0.0,
        float(bool(HTML_RE.search(text))),
        math.log1p(len(text)),
    ]


def describe_signals(text):
    """Readable {name: value} version, handy for checking a single email."""
    return dict(zip(FEATURE_NAMES, signals_for(text)))
