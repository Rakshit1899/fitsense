"""Footwear transfer test: does a model trained on apparel reviews read shoe reviews?

Step 1 (sample):  python -m fitsense.footwear sample --n 250
    Streams Amazon Reviews 2023 (Clothing, Shoes and Jewelry), keeps reviews that
    talk about shoes AND sizing, and writes data/footwear_to_label.csv.
Step 2 (label):   open the CSV in Excel and fill the "label" column with
    small / fit / large, or skip if the review does not say how it fits.
    Save it as data/footwear_labeled.csv. The model's predictions are not in
    the file, so they cannot influence your labels.
Step 3 (score):   python -m fitsense.footwear score
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, LABEL2ID
from .metrics import evaluate
from .predict import DEFAULT_TRANSFORMER, load_baseline, load_transformer

ROOT = Path(__file__).resolve().parents[2]
AMAZON_URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
              "review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz")

SHOE = re.compile(r"\b(shoes?|sneakers?|trainers?|running shoes?|boots?|loafers?|cleats?|"
                  r"slip[- ]ons?|high[- ]tops?)\b", re.I)
FIT = re.compile(r"\b(size[ds]?|sizing|fit|fits|fitting|runs? (small|large|big|narrow|wide)|"
                 r"tight|snug|loose|roomy|narrow|wide|half size|true to size|tts|toe box)\b", re.I)
# skip reviews that are about clothing that happen to mention shoes
NOT_SHOE = re.compile(r"\b(dress|shirt|jeans|pants|bra|jacket|hoodie|leggings|socks)\b", re.I)


def is_candidate(text: str) -> bool:
    return (40 <= len(text) <= 1200 and bool(SHOE.search(text)) and bool(FIT.search(text))
            and not NOT_SHOE.search(text))


def sample(n: int = 250, pool: int = 3000, max_lines: int = 3_000_000, seed: int = 7) -> Path:
    """Stream the file and stop early, so the full multi-GB download is never needed."""
    rows, seen = [], set()
    req = urllib.request.Request(AMAZON_URL, headers={"User-Agent": "fitsense"})
    with urllib.request.urlopen(req) as resp, gzip.GzipFile(fileobj=resp) as gz:
        for i, line in enumerate(io.TextIOWrapper(gz, encoding="utf-8")):
            if i >= max_lines or len(rows) >= pool:
                break
            r = json.loads(line)
            text = f"{r.get('title', '')}. {r.get('text', '')}".strip(". ").replace("<br />", " ")
            # one review per product keeps the test set diverse
            if r.get("parent_asin") in seen or not is_candidate(text):
                continue
            seen.add(r.get("parent_asin"))
            rows.append({"review_id": f"amz_{i}", "parent_asin": r.get("parent_asin"),
                         "rating": r.get("rating"), "text": text})
    print(f"scanned {i + 1:,} reviews, found {len(rows):,} shoe + sizing candidates")
    df = pd.DataFrame(rows).sample(min(n, len(rows)), random_state=seed)
    df["label"] = ""
    out = DATA_DIR / "footwear_to_label.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["review_id", "text", "label", "parent_asin", "rating"]].to_csv(out, index=False)
    print(f"wrote {len(df)} reviews to {out}")
    return out


def score(path: Path = DATA_DIR / "footwear_labeled.csv", out_dir: Path = ROOT / "reports") -> dict:
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABEL2ID)]
    print(f"{len(df)} labeled shoe reviews: " + ", ".join(
        f"{k} {v}" for k, v in df["label"].value_counts().items()))
    y = df["label"].map(LABEL2ID).to_numpy()

    results = {}
    models = [("baseline", load_baseline)]
    if DEFAULT_TRANSFORMER.exists():
        models.append(("distilbert", load_transformer))
    for name, loader in models:
        preds = np.asarray(loader()(df["text"].tolist())).argmax(1)
        results[name] = evaluate(y, preds, f"{name}_footwear", out_dir)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["sample", "score"])
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--file", type=Path, default=DATA_DIR / "footwear_labeled.csv")
    a = p.parse_args()
    sample(a.n) if a.step == "sample" else score(a.file)


if __name__ == "__main__":
    main()
