"""FastAPI service: score reviews and return a product-level fit verdict.

Run:  uvicorn app.api:app --reload
Uses the fine-tuned transformer if models/distilbert-base-uncased/best exists, else the baseline.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from fitsense.aggregate import item_fit_profile
from fitsense.data import LABELS
from fitsense.predict import load_best

app = FastAPI(title="FitSense", description="Does this product run small, true to size, or large?")
predict_proba, MODEL_KIND = load_best()


class Reviews(BaseModel):
    item_id: str = "item"
    reviews: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_KIND}


@app.post("/fit")
def fit(req: Reviews):
    probs = predict_proba(req.reviews)
    row = item_fit_profile([req.item_id] * len(req.reviews), probs, min_reviews=5).iloc[0]
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
