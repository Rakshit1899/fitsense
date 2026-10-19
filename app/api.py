"""FastAPI service: score reviews and return a product-level fit verdict.

Run:  uvicorn app.api:app --reload
Uses the fine-tuned transformer if models/<name>/best exists, else the baseline.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from fitsense.aggregate import item_fit_profile
from fitsense.data import LABELS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("FITSENSE_MODEL", ROOT / "models" / "distilbert-base-uncased" / "best"))

app = FastAPI(title="FitSense", description="Does this product run small, true to size, or large?")


def _load():
    if MODEL_DIR.exists():
        from transformers import pipeline
        clf = pipeline("text-classification", model=str(MODEL_DIR), top_k=None, truncation=True)

        def predict(texts):
            out = clf(texts)
            return np.array([[{d["label"]: d["score"] for d in row}[c] for c in LABELS] for row in out])
        return predict, "transformer"

    import joblib
    model = joblib.load(ROOT / "models" / "baseline.joblib")
    return model.predict_proba, "baseline"


predict_proba, MODEL_KIND = _load()


class Reviews(BaseModel):
    item_id: str = "item"
    reviews: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_KIND}


@app.post("/fit")
def fit(req: Reviews):
    probs = predict_proba(req.reviews)
    profile = item_fit_profile([req.item_id] * len(req.reviews), probs, min_reviews=5)
    row = profile.iloc[0]
    return {
        "item_id": req.item_id,
        "advice": row["advice"],
        "fit_skew": round(float(row["fit_skew"]), 3),
        "n_reviews": int(row["n_reviews"]),
        "review_predictions": [
            {"text": t[:120], "label": LABELS[int(p.argmax())], "confidence": round(float(p.max()), 3)}
            for t, p in zip(req.reviews, probs)
        ],
        "model": MODEL_KIND,
    }
