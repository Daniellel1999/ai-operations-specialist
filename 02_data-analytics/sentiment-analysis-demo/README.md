# sentiment-analysis-demo

Data analytics on a Hugging Face dataset, ending in sentiment analysis with
a pretrained model. No training or fine-tuning — this is EDA + inference.

## Run in Colab

Upload `colab_notebook.py` to Colab (File → Upload notebook, or paste cells
in) — the `# %%` markers split it into cells automatically. Alternatively,
open it locally in VS Code / Jupyter with the Jupytext or "Python Interactive"
cell format.

## Run locally

Dependencies are declared in the [root `pyproject.toml`](../../pyproject.toml). From the repo root:

```
uv sync
```

Then open `colab_notebook.py` in VS Code / Jupyter (see above) and skip the
`!pip install` and `google.colab` / Hugging Face login cells — those are
Colab-specific and not needed locally.

## Hugging Face token (optional)

Not required for this notebook — IMDB is a public dataset. Only needed for
gated/private datasets or models, or to raise anonymous rate limits.

In Colab: click the key icon in the left sidebar, add a secret named
`HF_TOKEN` with a token from https://huggingface.co/settings/tokens, then
run the login cell near the top of the notebook.

## Process

```
Hugging Face dataset (IMDB reviews)
 |
 v
Load into pandas
 |
 v
Explore: class balance, review length, missing values
 |
 v
Light cleaning (strip HTML artifacts)
 |
 v
Pretrained sentiment pipeline (DistilBERT / SST-2)
 |
 v
Accuracy vs ground-truth labels + confusion matrix
 |
 v
Inspect disagreements
```

## Steps

1. **Load** — `datasets.load_dataset("stanfordnlp/imdb")`, sampled down to 200 rows for speed.
2. **Explore** — label balance, review length distribution, missing values.
3. **Clean** — strip leftover `<br />` tags; transformer models need little else.
4. **Classify** — Hugging Face `pipeline("sentiment-analysis")` with
   `distilbert-base-uncased-finetuned-sst-2-english`.
5. **Evaluate** — accuracy against the dataset's real labels, plus a
   confusion matrix.
6. **Inspect** — pull out the reviews the model got wrong (sarcasm, mixed
   sentiment, etc.) as the most interesting part of the analysis.

## Scope

Uses a pretrained classifier as-is — no training or fine-tuning. Swap the
dataset name to reuse this notebook on Yelp, Amazon reviews, tweets, or any
other labeled Hugging Face text dataset.
