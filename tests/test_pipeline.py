"""Smoke tests on synthetic reviews, so the pipeline can be checked without the real data."""
import numpy as np
import pandas as pd

from fitsense.aggregate import item_fit_profile, label_skew
from fitsense.baseline import build
from fitsense.data import split_by_item
from fitsense.metrics import evaluate

PHRASES = {
    0: ["runs small, size up", "way too tight in the toe", "ordered my usual size and it was snug"],
    1: ["fits true to size", "perfect fit", "my normal size fit great"],
    2: ["runs large, size down", "too loose around the waist", "had to return for a smaller size"],
}


def fake_reviews(n_items=60, per_item=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        bias = rng.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
        for _ in range(per_item):
            lab = bias if rng.random() < 0.6 else rng.choice([0, 1, 2])
            rows.append({"item_id": f"i{i}", "label_id": lab,
                         "label": ["small", "fit", "large"][lab],
                         "text": f"{rng.choice(PHRASES[lab])}. nice color"})
    return pd.DataFrame(rows)


def test_split_keeps_items_together():
    df = fake_reviews()
    tr, va, te = split_by_item(df)
    assert not set(tr.item_id) & set(te.item_id)
    assert not set(tr.item_id) & set(va.item_id)
    assert len(tr) + len(va) + len(te) == len(df)


def test_baseline_learns_and_aggregates():
    df = fake_reviews()
    tr, _, te = split_by_item(df)
    model = build().set_params(tfidf__min_df=1).fit(tr["text"], tr["label_id"])
    res = evaluate(te["label_id"], model.predict(te["text"]), "smoke")
    assert res["macro_f1"] > 0.9

    profile = item_fit_profile(te["item_id"], model.predict_proba(te["text"]))
    truth = label_skew(te).reindex(profile["item_id"]).to_numpy()
    assert np.corrcoef(profile["fit_skew"], truth)[0, 1] > 0.9
    assert set(profile["advice"]) <= {"runs small: consider sizing up", "true to size",
                                      "runs large: consider sizing down", "not enough reviews"}
