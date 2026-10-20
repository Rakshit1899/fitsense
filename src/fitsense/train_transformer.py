"""Fine-tune a small transformer (DistilBERT by default) on fit labels.

Run on a GPU (Colab or Kaggle). Example:
    python -m fitsense.train_transformer --model distilbert-base-uncased --epochs 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from .data import LABEL2ID, LABELS, load_all, split_by_item
from .metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights, since "fit" dominates the labels."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def class_weights(labels) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(LABELS))
    w = len(labels) / (len(LABELS) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--sample", type=int, default=0, help="train on N rows for a quick dry run")
    args = p.parse_args()

    df = load_all()
    train, val, test = split_by_item(df)
    if args.sample:
        train = train.sample(min(args.sample, len(train)), random_state=0)

    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(frame):
        ds = Dataset.from_dict({"text": frame["text"].tolist(), "labels": frame["label_id"].tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])

    ds_train, ds_val, ds_test = to_ds(train), to_ds(val), to_ds(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)

    out = ROOT / "models" / args.model.rstrip("/\\").replace("\\", "/").split("/")[-1]
    # warmup as a step count works on both Transformers 4.x and 5.x
    total_steps = int(np.ceil(len(ds_train) / args.batch) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        warmup_steps=max(1, int(0.06 * total_steps)),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(pred):
        from sklearn.metrics import f1_score
        return {"macro_f1": f1_score(pred.label_ids, pred.predictions.argmax(-1), average="macro")}

    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        class_weights=class_weights(train["label_id"].to_numpy()),
    )
    trainer.train()

    preds = trainer.predict(ds_test).predictions.argmax(-1)
    evaluate(test["label_id"], preds, f"{out.name}_test", ROOT / "reports")
    trainer.save_model(str(out / "best"))
    tok.save_pretrained(str(out / "best"))


if __name__ == "__main__":
    main()
