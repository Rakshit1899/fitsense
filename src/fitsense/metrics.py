"""Shared evaluation so the baseline and the transformer are scored the same way."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import LABELS


def evaluate(y_true, y_pred, name: str, out_dir: Path | None = None) -> dict:
    """Macro F1 is the headline metric: most reviews say "fit", so accuracy
    alone would reward a model that never predicts small or large."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    result = {
        "model": name,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": classification_report(
            y_true, y_pred, labels=list(range(len(LABELS))),
            target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }
    print(f"\n{name}: macro F1 {result['macro_f1']:.3f} | accuracy {result['accuracy']:.3f}")
    print(classification_report(y_true, y_pred, labels=list(range(len(LABELS))),
                                target_names=LABELS, zero_division=0))
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2))
    return result
