"""
Used to score the model on real-world test emails.

The dataset test score shows how the model does on emails like its training data.
This shows how it does on the email it will actually meet.
"""
import pandas as pd

from email_model import Email
from ml.data import REAL_TEST_DIR, email_to_text
from ml.pipeline import OUTPUT_DIR, load_model, safety_score, text_fingerprint


def main():
    bundle = load_model()
    pipeline, threshold = bundle["pipeline"], bundle["threshold"]
    seen_in_training = bundle.get("train_fingerprints", set())
    print(f"Model trained {bundle['trained_at']}, threshold {threshold}\n")

    rows = []
    for folder, expected in (("legit", "safe"), ("phishing", "PHISHING")):
        for path in sorted((REAL_TEST_DIR / folder).glob("*.eml")):
            email = Email(path.name, path.read_bytes())
            text = email_to_text(email)
            probability = pipeline.predict_proba([text])[0, 1]
            verdict = "PHISHING" if probability >= threshold else "safe"
            rows.append({
                "folder": folder, "file": path.name, "expected": expected, "verdict": verdict,
                "correct": verdict == expected, "safety_score": safety_score(probability),
                "in_training_data": text_fingerprint(text) in seen_in_training,
                "subject": email.subject[:70],
            })

    if not rows:
        print(f"No .eml files found. Put some in {REAL_TEST_DIR / 'legit'} and {REAL_TEST_DIR / 'phishing'}.")
        return

    df = pd.DataFrame(rows)
    for folder, group in df.groupby("folder", sort=False):
        print(f"=== {folder} ({len(group)} emails) ===")
        for _, r in group.sort_values("safety_score").iterrows():
            mark = "  " if r["correct"] else "X "
            print(f"{mark}{r['safety_score']:>3}  {r['verdict']:<8}  {r['file'][:40]:<40}  {r['subject']}")
        print(f"correct: {group['correct'].sum()} of {len(group)} ({group['correct'].mean():.0%}), "
              f"median safety score {group['safety_score'].median():.0f}\n")

    reused = int(df["in_training_data"].sum())
    if reused:
        print(f"WARNING: {reused} of these emails were also in the training data, so their scores are "
              "too optimistic. Remove them from the training sources (or from real_test) and retrain.\n")

    print(f"Overall: {df['correct'].sum()} of {len(df)} correct ({df['correct'].mean():.0%}). "
          "X marks the wrong ones.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "real_test_results.csv", index=False)
    print(f"Saved to {OUTPUT_DIR / 'real_test_results.csv'}")


if __name__ == "__main__":
    main()
