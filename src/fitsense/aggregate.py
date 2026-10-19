"""Turn review-level predictions into a product-level fit signal.

This is the part a retailer would actually use: "this item runs small,
consider sizing up", backed by how many reviews say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LABELS


def item_fit_profile(item_ids, probs: np.ndarray, min_reviews: int = 10,
                     threshold: float = 0.15) -> pd.DataFrame:
    """probs: (n_reviews, 3) class probabilities in LABELS order (small, fit, large).

    fit_skew = mean P(large) - mean P(small), in [-1, 1].
    Negative means the item runs small, positive means it runs large.
    Items with fewer than min_reviews get "not enough reviews" rather than a guess.
    """
    frame = pd.DataFrame(probs, columns=[f"p_{c}" for c in LABELS])
    frame["item_id"] = list(item_ids)
    g = frame.groupby("item_id")
    out = g[[f"p_{c}" for c in LABELS]].mean()
    out["n_reviews"] = g.size()
    out["fit_skew"] = out["p_large"] - out["p_small"]

    # standard error of the skew, so a small sample does not look certain
    per_review = frame["p_large"] - frame["p_small"]
    out["skew_se"] = per_review.groupby(frame["item_id"]).std(ddof=1) / np.sqrt(out["n_reviews"])

    def verdict(row):
        if row.n_reviews < min_reviews:
            return "not enough reviews"
        if row.fit_skew <= -threshold and row.fit_skew + 2 * row.skew_se < 0:
            return "runs small: consider sizing up"
        if row.fit_skew >= threshold and row.fit_skew - 2 * row.skew_se > 0:
            return "runs large: consider sizing down"
        return "true to size"

    out["advice"] = out.apply(verdict, axis=1)
    return out.reset_index().sort_values("fit_skew")


def label_skew(df: pd.DataFrame) -> pd.Series:
    """Ground-truth skew per item from the real fit labels, used to check the
    predicted skew on held-out items."""
    share = pd.crosstab(df["item_id"], df["label"], normalize="index")
    for c in LABELS:
        if c not in share:
            share[c] = 0.0
    return share["large"] - share["small"]
