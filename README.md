# FitSense: turning reviews into a size and fit signal

Fit is one of the most common reasons apparel and footwear get returned, and the answer is often already sitting in the reviews ("runs small", "size up half a size"). Most retailers never turn that text into a structured signal. FitSense reads reviews, classifies each one as **small / fit / large**, and rolls them up into product-level advice like *"runs small: consider sizing up"*, with a guard against calling it from too few reviews.

## Data

UCSD Clothing Fit datasets (Misra, Wan, McAuley, RecSys 2018). Every review has a real fit label, so no hand-labeling is needed.

| Source | Reviews | Items | Fields used |
|---|---|---|---|
| RentTheRunway | ~192K | ~5.9K | review text, summary, fit, size, category |
| ModCloth | ~83K | ~1.4K | review text, summary, fit, size, category |

`src/fitsense/data.py` downloads both on first run.

## Approach

1. **Split by item, not by row.** All reviews for a product go to the same split, so the model can't memorize product-specific wording.
2. **Baseline:** TF-IDF + logistic regression with balanced class weights.
3. **Transformer:** fine-tune DistilBERT (then try DeBERTa-v3-small) with class-weighted loss, since most reviews say "fit".
4. **Product-level roll-up:** fit skew = mean P(large) minus mean P(small). Advice is only given when the skew is clear of noise (2 standard errors) and the item has enough reviews.
5. **Serve:** FastAPI endpoint that takes a product's reviews and returns the verdict.

## Evaluation

| Level | Metric | Why |
|---|---|---|
| Review | Macro F1, per-class recall | Accuracy alone rewards always predicting "fit" |
| Product | Correlation of predicted vs true skew on held-out items | What a retailer actually acts on |
| Product | Share of items flagged correctly as small / large | Precision of the advice itself |
| Domain shift | ~200 hand-labeled Amazon shoe reviews | Trained on apparel, does it transfer to footwear? |

## Results

**Review level** (held-out test set, 40,775 reviews from products not seen in training)

| Model | Macro F1 | Accuracy | Small F1 | Fit F1 | Large F1 |
|---|---|---|---|---|---|
| TF-IDF + LR | 0.682 | 0.761 | 0.62 | 0.84 | 0.59 |
| DistilBERT | | | | | |

**Product level** (held-out products with 10+ reviews)

| Model | Skew correlation (Spearman) | Advice accuracy | Flag precision | Flag recall |
|---|---|---|---|---|
| TF-IDF + LR | 0.921 | 0.776 | 0.884 | 0.660 |
| DistilBERT | | | | |

*Flag precision:* of products the model said run small or large, the share that really did. *Flag recall:* of products that really ran small or large, the share it caught. The advice is deliberately conservative (it needs the skew to clear 2 standard errors), so precision is favored over recall: a wrong "size up" costs a return, a missed one leaves the shopper where they started.

**Footwear transfer** (hand-labeled Amazon shoe reviews, model trained only on apparel)

| Model | Macro F1 | Accuracy |
|---|---|---|
| TF-IDF + LR | | |
| DistilBERT | | |

## Run it

```bash
pip install -r requirements.txt
python -m pytest                              # smoke tests on synthetic data
PYTHONPATH=src python -m fitsense.data        # download + label counts
PYTHONPATH=src python -m fitsense.baseline    # baseline (CPU, a few minutes)
PYTHONPATH=src python -m fitsense.train_transformer --sample 5000 --epochs 1   # dry run
PYTHONPATH=src python -m fitsense.train_transformer --epochs 2                 # full run, GPU
PYTHONPATH=src python -m fitsense.evaluate_items  # product-level evaluation
PYTHONPATH=src python -m fitsense.footwear sample # shoe reviews to hand-label
PYTHONPATH=src python -m fitsense.footwear score  # score both models on labeled shoes
PYTHONPATH=src uvicorn app.api:app --reload   # API at http://127.0.0.1:8000/docs
```

**GPU:** use Colab (Runtime > Change runtime type > T4 GPU) or Kaggle notebooks. Full DistilBERT training on ~270K reviews takes roughly 1 to 2 hours on a T4. Start with `--sample` to check everything runs. `FitSense_Colab.ipynb` runs the whole pipeline on Colab, top to bottom.

## Project structure

```
src/fitsense/
  data.py               download, clean, split by item
  baseline.py           TF-IDF + logistic regression
  train_transformer.py  DistilBERT fine-tuning with class weights
  metrics.py            shared evaluation (macro F1, confusion matrix)
  aggregate.py          review predictions -> product fit advice
  predict.py            shared inference for baseline and transformer
  evaluate_items.py     product-level evaluation on held-out products
  footwear.py           Amazon shoe review sampling + transfer test
app/api.py              FastAPI service
FitSense_Colab.ipynb    end-to-end GPU run on Colab
tests/                  smoke tests on synthetic reviews
```

## Citation

Rishabh Misra, Mengting Wan, Julian McAuley. *Decomposing fit semantics for product size recommendation in metric spaces.* RecSys 2018.
