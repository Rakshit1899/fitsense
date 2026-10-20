"""Product-level evaluation on held-out products.

Review-level F1 says how often a single review is read correctly. A retailer acts
on the product verdict, so this checks whether the rolled-up advice matches what
customers actually reported for products the model never saw.

    python -m fitsense.evaluate_items            # baseline and transformer if trained
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .aggregate import item_fit_profile, label_skew
from .data import load_all, split_by_item
from .predict import DEFAULT_TRANSFORMER, load_baseline, load_transformer

ROOT = Path(__file__).resolve().parents[2]
MIN_REVIEWS = 10
THRESHOLD = 0.15


def true_direction(skew: float) -> str:
    if skew <= -THRESHOLD:
        return "small"
    if skew >= THRESHOLD:
        return "large"
    return "true to size"


def advice_direction(advice: str) -> str:
    return {"runs small: consider sizing up": "small",
            "runs large: consider sizing down": "large"}.get(advice, "true to size")


def evaluate_products(test: pd.DataFrame, predict_proba, name: str, out_dir: Path | None = None) -> dict:
    probs = predict_proba(test["text"].tolist())
    prof = item_fit_profile(test["item_id"], probs, min_reviews=MIN_REVIEWS, threshold=THRESHOLD)
    prof["true_skew"] = label_skew(test).reindex(prof["item_id"]).to_numpy()

    rated = prof[prof["n_reviews"] >= MIN_REVIEWS].copy()
    rated["true_dir"] = rated["true_skew"].map(true_direction)
    rated["pred_dir"] = rated["advice"].map(advice_direction)
    flagged = rated[rated["pred_dir"] != "true to size"]
    truly_off = rated[rated["true_dir"] != "true to size"]

    result = {
        "model": name,
        "products_scored": int(len(rated)),
        "products_skipped_few_reviews": int((prof["n_reviews"] < MIN_REVIEWS).sum()),
        "skew_pearson": float(pearsonr(rated["fit_skew"], rated["true_skew"])[0]),
        "skew_spearman": float(spearmanr(rated["fit_skew"], rated["true_skew"])[0]),
        "advice_accuracy": float((rated["pred_dir"] == rated["true_dir"]).mean()),
        "products_flagged": int(len(flagged)),
        # of products we told shoppers to size up or down, how many really ran that way
        "flag_precision": float((flagged["pred_dir"] == flagged["true_dir"]).mean()) if len(flagged) else float("nan"),
        # of products that really ran small or large, how many we caught
        "flag_recall": float((truly_off["pred_dir"] == truly_off["true_dir"]).mean()) if len(truly_off) else float("nan"),
    }
    print(f"\n{name} (product level, {result['products_scored']} products with {MIN_REVIEWS}+ reviews)")
    for k in ("skew_pearson", "skew_spearman", "advice_accuracy", "flag_precision", "flag_recall"):
        print(f"  {k:16s} {result[k]:.3f}")
    print(f"  products flagged {result['products_flagged']}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}_product_metrics.json").write_text(json.dumps(result, indent=2))
        prof.to_csv(out_dir / f"{name}_product_advice.csv", index=False)
    return result


def main():
    _, _, test = split_by_item(load_all())
    models = [("baseline", load_baseline)]
    if DEFAULT_TRANSFORMER.exists():
        models.append(("distilbert", load_transformer))
    for name, loader in models:
        evaluate_products(test, loader(), name, ROOT / "reports")


if __name__ == "__main__":
    main()
