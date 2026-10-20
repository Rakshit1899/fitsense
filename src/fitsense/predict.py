"""One way to get class probabilities from either model, used by evaluation and the API."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import LABELS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSFORMER = ROOT / "models" / "distilbert-base-uncased" / "best"


def load_baseline(path: Path = ROOT / "models" / "baseline.joblib"):
    import joblib
    model = joblib.load(path)
    return lambda texts: model.predict_proba(list(texts))


def load_transformer(path: Path = DEFAULT_TRANSFORMER, batch_size: int = 128, max_len: int = 256):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device).eval()
    # map the model's label order onto LABELS order (small, fit, large)
    order = [model.config.label2id[c] for c in LABELS]

    @torch.no_grad()
    def predict_proba(texts):
        texts, out = list(texts), []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, dim=-1)[:, order].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))

    return predict_proba


def load_best():
    """Transformer if it has been trained, otherwise the baseline."""
    if DEFAULT_TRANSFORMER.exists():
        return load_transformer(), "transformer"
    return load_baseline(), "baseline"
