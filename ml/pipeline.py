"""
Shared pieces used by the training scripts: building the model, evaluating it,
and saving it.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from ml.features import EmailSignals

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

ML_DIR = Path(__file__).parent
OUTPUT_DIR = ML_DIR / "output"
MODEL_PATH = ML_DIR / "models" / "phishing_model.joblib"
RANDOM_STATE = 42  # fixed, so every run splits the data the same way and results are comparable

# Every model here can give a probability (predict_proba), which becomes the safety score.
MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "naive_bayes": lambda: MultinomialNB(alpha=0.1),
    # LinearSVC has no probabilities on its own; calibration adds them
    "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE),
}


def build_pipeline(model="logistic_regression", use_signals=True):
    """
    Text in, phishing probability out:
      1. TF-IDF turns the text into word and two-word-phrase scores
      2. EmailSignals adds the hand-made signals (optional)
      3. the classifier learns from both
    """
    words = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),     # single words and two-word phrases ("verify your")
        min_df=2,               # ignore words seen in only one email
        max_df=0.95,            # ignore words in almost every email
        max_features=100_000,   # cap the vocabulary size
        sublinear_tf=True,      # a word repeated 50 times isn't 50x as important
    )
    parts = [("words", words)]
    if use_signals:
        parts.append(("signals", Pipeline([("extract", EmailSignals()), ("scale", MaxAbsScaler())])))
    return Pipeline([("features", FeatureUnion(parts)), ("clf", MODELS[model]())])


def safety_score(phishing_probability):
    """0-100, where 100 is safest: the opposite of the phishing probability."""
    return int(round(100 * (1 - float(phishing_probability))))


def evaluate(pipeline, test_df, threshold=0.5, out_dir=OUTPUT_DIR):
    """
    Score the model on emails it never saw during training. Prints a report and
    writes metrics.json, confusion_matrix.png, mistakes.csv and (for linear
    models) top_features.txt to ml/output/.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = test_df["label"].to_numpy()
    proba = pipeline.predict_proba(test_df["text"])[:, 1]
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "threshold": threshold,
        "test_emails": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }

    print("\n=== Test set results ===")
    print(classification_report(y_true, y_pred, target_names=["legitimate", "phishing"], digits=3))
    print(f"Phishing emails missed (false negatives): {fn}")
    print(f"Safe emails wrongly flagged (false positives): {fp}")
    print(f"ROC AUC: {metrics['roc_auc']}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["legitimate", "phishing"],
                                            cmap="Blues", values_format="d")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Every email the model got wrong, most confident mistakes first: the best place to look for ideas.
    mistakes = test_df.assign(phishing_probability=proba.round(4), predicted=y_pred)
    mistakes = mistakes[mistakes["predicted"] != mistakes["label"]].copy()
    mistakes["confidence_in_wrong_answer"] = np.where(
        mistakes["predicted"] == 1, mistakes["phishing_probability"], 1 - mistakes["phishing_probability"])
    mistakes["text"] = mistakes["text"].str.slice(0, 600)
    mistakes.sort_values("confidence_in_wrong_answer", ascending=False).to_csv(out_dir / "mistakes.csv", index=False)

    write_top_features(pipeline, out_dir / "top_features.txt")
    print(f"\nSaved metrics, chart and mistakes to {out_dir}")
    return metrics


def write_top_features(pipeline, path, count=25):
    """For logistic regression: the words and signals that push hardest toward each answer."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return
    names = pipeline.named_steps["features"].get_feature_names_out()
    weights = clf.coef_[0]
    order = np.argsort(weights)
    lines = ["Strongest signs of PHISHING:"]
    lines += [f"  {weights[i]:+.3f}  {names[i]}" for i in order[::-1][:count]]
    lines += ["", "Strongest signs of LEGITIMATE email:"]
    lines += [f"  {weights[i]:+.3f}  {names[i]}" for i in order[:count]]
    path.write_text("\n".join(lines))


def text_fingerprint(text):
    """Short, fixed-length ID for an email's text, used to spot training emails reused for testing."""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def save_model(pipeline, threshold, metrics, settings, train_texts=()):
    """Save the trained pipeline plus everything needed to use it later."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "metrics": metrics,
        "settings": settings,
        "train_fingerprints": {text_fingerprint(t) for t in train_texts},
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1_000_000
    print(f"Saved model to {MODEL_PATH} ({size_mb:.1f} MB)")


def load_model():
    return joblib.load(MODEL_PATH)
