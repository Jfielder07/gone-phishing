"""
Model tuning script, used for adjusting settings to determine the best ones for the model. 

Run from the project folder (this takes longer than train_model.py):
    python -m ml.tune_model
    python -m ml.tune_model --target-recall 0.97 --sample 10000
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from ml.data import load_emails
from ml.pipeline import RANDOM_STATE, build_pipeline, evaluate, save_model

# Settings to try. Every combination is tested, so each extra value multiplies the run time.
PARAM_GRID = {
    "features__words__ngram_range": [(1, 1), (1, 2)],  # single words, or words + two-word phrases
    "features__words__min_df": [2, 5],                  # how rare a word can be and still count
    "clf__C": [0.3, 1, 3, 10],                          # higher C = model fits training data more closely
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tune the Gone Phishing model.")
    parser.add_argument("--target-recall", type=float, default=0.95,
                        help="share of phishing emails the model must catch (default 0.95)")
    parser.add_argument("--sample", type=int, default=0, help="use only this many emails")
    return parser.parse_args()


def choose_threshold(y_true, proba, target_recall):
    """
    Lowering the threshold catches more phishing (higher recall) but flags more
    safe emails too (lower precision). Among the thresholds that reach the target
    recall, pick the most precise one; if several tie, the one that catches more phishing.
    """
    rows = []
    for t in np.round(np.arange(0.20, 0.81, 0.05), 2):
        pred = (proba >= t).astype(int)
        rows.append({"threshold": t,
                     "precision": precision_score(y_true, pred, zero_division=0),
                     "recall": recall_score(y_true, pred, zero_division=0),
                     "f1": f1_score(y_true, pred, zero_division=0)})
    table = pd.DataFrame(rows)
    print("\n=== Threshold options (cross-validated on training data) ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    good = table[table["recall"] >= target_recall]
    if good.empty:
        print(f"\nNo threshold reaches recall {target_recall}; using the one with the best F1.")
        return float(table.loc[table["f1"].idxmax(), "threshold"])
    best = good.sort_values(["precision", "recall"], ascending=False).iloc[0]
    chosen = float(best["threshold"])
    print(f"\nChosen threshold: {chosen} (most precise option with recall >= {target_recall}: "
          f"precision {best['precision']:.3f}, recall {best['recall']:.3f})")
    return chosen


def main():
    args = parse_args()
    df, report = load_emails()
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=RANDOM_STATE)

    # Same split as train_model.py (same random_state), so the test results are comparable.
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE)
    X_train, y_train = train_df["text"], train_df["label"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    combos = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    print(f"Trying {combos} combinations x 5 folds = {combos * 5} training runs on {len(train_df):,} emails...")
    search = GridSearchCV(build_pipeline("logistic_regression"), PARAM_GRID, scoring="f1",
                          cv=folds, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values("rank_test_score")[["params", "mean_test_score", "std_test_score"]].head(8)
    print("\n=== Best settings (F1, cross-validated) ===")
    for _, row in results.iterrows():
        print(f"  F1 {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})  {row['params']}")
    print(f"\nBest: {search.best_params_}")

    # Out-of-fold probabilities: each training email is scored by a model that didn't train on it.
    proba = cross_val_predict(search.best_estimator_, X_train, y_train, cv=folds,
                              method="predict_proba", n_jobs=-1)[:, 1]
    threshold = choose_threshold(y_train.to_numpy(), proba, args.target_recall)

    # The test set is used once, at the very end.
    metrics = evaluate(search.best_estimator_, test_df, threshold)
    save_model(search.best_estimator_, threshold, metrics, {
        "model": "logistic_regression", "use_signals": True, "best_params": search.best_params_,
        "target_recall": args.target_recall, "train_emails": len(train_df),
        "test_emails": len(test_df), "data_report": report,
    }, train_texts=train_df["text"])


if __name__ == "__main__":
    main()
