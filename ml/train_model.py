"""
Step 2: compare models, train the best one, test it, and save it.

Run from the project folder:
    python -m ml.train_model                          # compare all, then train logistic regression
    python -m ml.train_model --model naive_bayes      # train a different model
    python -m ml.train_model --no-signals             # words only, to see if the signals help
    python -m ml.train_model --skip-compare           # just train and test (faster)
    python -m ml.train_model --sample 5000            # use 5,000 emails for a quick trial run
"""
import argparse
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from ml.data import load_emails
from ml.pipeline import MODELS, RANDOM_STATE, build_pipeline, evaluate, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Gone Phishing model.")
    parser.add_argument("--model", default="logistic_regression", choices=MODELS)
    parser.add_argument("--no-signals", action="store_true", help="use word features only")
    parser.add_argument("--skip-compare", action="store_true", help="don't compare models first")
    parser.add_argument("--include-forest", action="store_true", help="include random forest in the comparison (slow)")
    parser.add_argument("--sample", type=int, default=0, help="use only this many emails (quick test runs)")
    return parser.parse_args()


def compare_models(X_train, y_train, use_signals, include_forest):
    """5-fold cross-validation on the TRAINING data only. The test set stays untouched."""
    names = [m for m in MODELS if include_forest or m != "random_forest"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name in names:
        start = time.time()
        print(f"Cross-validating {name}...", flush=True)
        scores = cross_validate(build_pipeline(name, use_signals), X_train, y_train, cv=folds,
                                scoring=["accuracy", "precision", "recall", "f1"], n_jobs=-1)
        rows.append({
            "model": name,
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "f1_spread": scores["test_f1"].std(),
            "seconds": time.time() - start,
        })
    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print("\n=== Model comparison (5-fold cross-validation on training data) ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nHigher is better. f1_spread is how much F1 varied between folds: smaller means more reliable.")


def main():
    args = parse_args()
    use_signals = not args.no_signals

    df, report = load_emails()
    print("Data:", ", ".join(f"{k}={v:,}" for k, v in report.items()))
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=RANDOM_STATE)
        print(f"Using a random sample of {args.sample:,} emails")

    # Hold back 20% as the test set. stratify keeps the phishing/legitimate mix the same in both parts.
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE)
    print(f"Training on {len(train_df):,} emails, testing on {len(test_df):,}\n")

    if not args.skip_compare:
        compare_models(train_df["text"], train_df["label"], use_signals, args.include_forest)

    print(f"\nTraining {args.model} on all training data...", flush=True)
    start = time.time()
    pipeline = build_pipeline(args.model, use_signals)
    pipeline.fit(train_df["text"], train_df["label"])
    print(f"Trained in {time.time() - start:.1f} seconds")

    threshold = 0.5
    metrics = evaluate(pipeline, test_df, threshold)
    save_model(pipeline, threshold, metrics, {
        "model": args.model, "use_signals": use_signals,
        "train_emails": len(train_df), "test_emails": len(test_df), "data_report": report,
    }, train_texts=train_df["text"])


if __name__ == "__main__":
    main()
