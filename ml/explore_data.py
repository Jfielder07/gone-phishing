"""
Step 1: look at the data before training on it.

Run from the project folder:
    python -m ml.explore_data
"""
from ml.data import DATA_DIR, PHISHING, SOURCES, load_emails


def preview(text, length=200):
    return " ".join(str(text).split())[:length]


def main():
    print(f"Looking in {DATA_DIR}\n")
    df, report = load_emails()

    # 1. Each source: how many emails, and one example of each label to confirm the labels make sense
    print("\n=== Each source ===")
    for name, use, limit in SOURCES:
        part = df[df["source"] == name]
        if part.empty:
            continue
        cap = f", capped at {limit:,}" if limit else ""
        print(f"\n--- {name} (using: {use}{cap}) ---")
        print("emails after cleaning:", part["label"].map({1: "phishing", 0: "legitimate"}).value_counts().to_dict())
        for label in sorted(part["label"].unique()):
            example = part[part["label"] == label].iloc[0]["text"]
            print(f"  example {'PHISHING' if label == PHISHING else 'LEGITIMATE'}: {preview(example)}")

    # 2. Everything combined and cleaned, as the training scripts will see it
    print("\n=== Combined and cleaned ===")
    for key, value in report.items():
        print(f"{key}: {value:,}")
    counts = df["label"].value_counts()
    phishing, legit = counts.get(PHISHING, 0), counts.get(1 - PHISHING, 0)
    print(f"\nphishing: {phishing:,}   legitimate: {legit:,}   ({phishing / len(df):.0%} phishing)")
    print("\nemails per source and label:")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    print("\nmedian length in characters:")
    print(df.assign(length=df["text"].str.len()).groupby("label")["length"].median())


if __name__ == "__main__":
    main()
