"""Download, load and split the UCSD Clothing Fit datasets (RentTheRunway, ModCloth).

Source: Misra, Wan, McAuley. "Decomposing fit semantics for product size
recommendation in metric spaces." RecSys 2018.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URLS = {
    "renttherunway": "https://mcauleylab.ucsd.edu/public_datasets/data/renttherunway/renttherunway_final_data.json.gz",
    "modcloth": "https://mcauleylab.ucsd.edu/public_datasets/data/modcloth/modcloth_final_data.json.gz",
}

LABELS = ["small", "fit", "large"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def download(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download one dataset if it is not already on disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}.json.gz"
    if not path.exists():
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(URLS[name], path)
    return path


def load(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load one dataset into a clean frame with columns:
    source, item_id, user_id, text, label, label_id, size, category.
    """
    df = pd.read_json(download(name, data_dir), lines=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    summary = df.get("review_summary", pd.Series("", index=df.index)).fillna("")
    body = df.get("review_text", pd.Series("", index=df.index)).fillna("")
    df["text"] = (summary.str.strip() + ". " + body.str.strip()).str.strip(". ").str.strip()

    df["label"] = df["fit"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABELS) & (df["text"].str.len() > 10)].copy()
    df["label_id"] = df["label"].map(LABEL2ID)
    df["source"] = name

    for col in ("size", "category"):
        if col not in df.columns:
            df[col] = np.nan
    return df[["source", "item_id", "user_id", "text", "label", "label_id", "size", "category"]]


def load_all(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = [load(name, data_dir) for name in URLS]
    df = pd.concat(frames, ignore_index=True)
    # item ids can collide across sources, so make them unique
    df["item_id"] = df["source"] + "_" + df["item_id"].astype(str)
    return df


def split_by_item(df: pd.DataFrame, val: float = 0.15, test: float = 0.15, seed: int = 42):
    """Split so that every review of an item lands in the same split.

    A random row split would leak item-specific wording (product names,
    "this dress") from train into test and overstate accuracy.
    """
    items = np.random.default_rng(seed).permutation(np.asarray(df["item_id"].unique(), dtype=object))
    n_test = int(len(items) * test)
    n_val = int(len(items) * val)
    test_items = set(items[:n_test])
    val_items = set(items[n_test:n_test + n_val])

    split = np.where(df["item_id"].isin(test_items), "test",
                     np.where(df["item_id"].isin(val_items), "val", "train"))
    return df[split == "train"], df[split == "val"], df[split == "test"]


if __name__ == "__main__":
    data = load_all()
    print(data.groupby(["source", "label"]).size().unstack())
    tr, va, te = split_by_item(data)
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,}")
