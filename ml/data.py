"""
Loading and cleaning the training data.

Three kinds of source can be used, all kept in the ml/data/ folder:

  CSV files   from https://zenodo.org/records/8339691
              Columns include subject, body and label (1 = phishing/spam, 0 = legitimate).
  .eml folder a folder of email files, e.g. the phishing_pot collection
              (https://github.com/rf-peixoto/phishing_pot or your own emails.
  .mbox file  a mailbox export, e.g. from Google Takeout or Thunderbird.

Emails read from .eml folders and .mbox files go through the same Email class
the website uses, so the model trains on text built exactly like the text it
will see later.
"""
import mailbox
import random
from pathlib import Path

import pandas as pd

from email_model import Email
from ml.text import MAX_CHARS, build_text, email_to_text  # noqa: F401  (re-exported for the scripts)

DATA_DIR = Path(__file__).parent / "data"
REAL_TEST_DIR = DATA_DIR / "real_test"  # never used for training: see evaluate_real.py
PHISHING = 1
LEGITIMATE = 0
RANDOM_STATE = 42

# Which sources to train on. Each line is:
#   (name inside ml/data,  what the emails are,  most emails to take or None for all)
#
# What the emails are:
#   "all"       CSV only: keep every row with its own label
#   "phishing"  CSV: keep only label 1 rows. Folder/.mbox: every email counts as phishing
#   "legit"     CSV: keep only label 0 rows. Folder/.mbox: every email counts as legitimate
#
# Some CSVs label SPAM (ads, newsletters) as 1, not just phishing, so only their
# legitimate rows are used. Sources that aren't in ml/data yet are skipped with a note.
SOURCES = [
    ("Nazario.csv",        "all",      None),   # phishing emails (older)
    ("Nigerian_Fraud.csv", "all",      None),   # advance-fee scams
    ("Enron.csv",          "legit",    8000),   # business email from 2001; capped so it doesn't dominate
    ("SpamAssasin.csv",    "legit",    None),   # personal and mailing-list email (misspelled on purpose)
    ("phishing_pot",       "phishing", 8000),   # folder of recent real phishing .eml files
    ("my_inbox.mbox",      "legit",    None),   # your own email, exported as a mailbox file
    ("my_legit_emails",    "legit",    None),   # your own email, as a folder of .eml files
]


def _pick(items, limit):
    """Take a random (but repeatable) selection of at most `limit` items."""
    items = list(items)
    if limit and len(items) > limit:
        items = random.Random(RANDOM_STATE).sample(items, limit)
    return items


def load_csv(path, use, limit):
    df = pd.read_csv(path, encoding_errors="replace", low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"subject", "body", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} has no {sorted(missing)} column. Columns found: {list(df.columns)}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([PHISHING, LEGITIMATE])].copy()
    df["label"] = df["label"].astype(int)
    if use == "phishing":
        df = df[df["label"] == PHISHING]
    elif use == "legit":
        df = df[df["label"] == LEGITIMATE]
    elif use != "all":
        raise ValueError(f"Unknown setting {use!r} for {path.name}")
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=RANDOM_STATE)

    texts = [build_text(s, b) for s, b in zip(df["subject"].fillna(""), df["body"].fillna(""))]
    return pd.DataFrame({"text": texts, "label": df["label"].to_numpy()})


def _emails_to_frame(named_bytes, label, source):
    """Parse (name, raw bytes) pairs with the Email class. Unreadable emails are skipped."""
    texts, skipped = [], 0
    for name, raw in named_bytes:
        try:
            texts.append(email_to_text(Email(name, raw)))
        except Exception:
            skipped += 1
    if skipped:
        print(f"  {source}: skipped {skipped} unreadable emails")
    return pd.DataFrame({"text": texts, "label": label})


def read_eml_folder(folder, limit=None):
    """Yield (name, bytes) for the .eml files in a folder and its subfolders."""
    for path in _pick(sorted(folder.rglob("*.eml")), limit):
        yield path.name, path.read_bytes()


def read_mbox(path, limit=None):
    """Yield (name, bytes) for the messages in a .mbox file, skipping Gmail's Spam and Trash."""
    box = mailbox.mbox(str(path), create=False)
    try:
        for key in _pick(box.keys(), limit):
            raw = box.get_bytes(key)
            labels = mailbox.mboxMessage(raw).get("X-Gmail-Labels", "")
            if "Spam" in labels or "Trash" in labels:
                continue
            yield f"{path.name}#{key}", raw
    finally:
        box.close()


def load_source(name, use, limit=None):
    """Load one source. Returns a DataFrame (text, label, source), or None if it isn't there."""
    path = DATA_DIR / name
    if not path.exists():
        return None
    print(f"Reading {name}...", flush=True)
    if path.is_file() and path.suffix.lower() == ".csv":
        df = load_csv(path, use, limit)
    else:
        if use not in ("phishing", "legit"):
            raise ValueError(f"{name}: email folders and .mbox files need 'phishing' or 'legit', not {use!r}")
        label = PHISHING if use == "phishing" else LEGITIMATE
        emails = read_mbox(path, limit) if path.is_file() else read_eml_folder(path, limit)
        df = _emails_to_frame(emails, label, name)
    df["source"] = name
    return df[["text", "label", "source"]]


def clean(df):
    """Remove empty emails and duplicates. Returns (cleaned DataFrame, report dict)."""
    report = {"rows_loaded": len(df)}

    df = df[df["text"].str.len() > 0]
    report["empty_removed"] = report["rows_loaded"] - len(df)

    # The same text with BOTH labels can't be learned from: drop it entirely.
    labels_per_text = df.groupby("text")["label"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    before = len(df)
    df = df[~df["text"].isin(conflicting)]
    report["conflicting_removed"] = before - len(df)

    # Exact duplicates would let the same email land in both the training and test
    # sets, which makes the test score look better than it really is.
    before = len(df)
    df = df.drop_duplicates(subset="text")
    report["duplicates_removed"] = before - len(df)

    report["rows_kept"] = len(df)
    return df.reset_index(drop=True), report


def real_test_texts():
    """Text of every email in the real-world test folders, which must never be trained on."""
    texts = set()
    for name, raw in read_eml_folder(REAL_TEST_DIR) if REAL_TEST_DIR.exists() else []:
        try:
            texts.add(email_to_text(Email(name, raw)))
        except Exception:
            pass
    return texts


def load_emails(sources=None):
    """Load every available source in SOURCES, combine and clean them."""
    frames = []
    for name, use, limit in sources or SOURCES:
        df = load_source(name, use, limit)
        if df is None:
            print(f"  (skipping {name}: not in ml/data)")
        else:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No training data found in {DATA_DIR}")
    df, report = clean(pd.concat(frames, ignore_index=True))

    # An inbox export usually contains the same emails saved for the test set.
    # Training on them would make the real-world test score meaningless, so drop them.
    held_out = real_test_texts()
    before = len(df)
    df = df[~df["text"].isin(held_out)].reset_index(drop=True)
    report["real_test_emails_excluded"] = before - len(df)
    report["rows_kept"] = len(df)
    return df, report
