"""TF-IDF + logistic regression baseline. The transformer has to beat this to earn its cost."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import load_all, split_by_item
from .metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]


def build() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                                  sublinear_tf=True)),
        # balanced weights so the rare small/large classes are not ignored
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def run(df=None, out_dir: Path = ROOT / "reports", model_dir: Path = ROOT / "models"):
    df = load_all() if df is None else df
    train, val, test = split_by_item(df)
    model = build().fit(train["text"], train["label_id"])
    evaluate(val["label_id"], model.predict(val["text"]), "baseline_val")
    result = evaluate(test["label_id"], model.predict(test["text"]), "baseline_test", out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "baseline.joblib")
    return model, result


if __name__ == "__main__":
    run()
