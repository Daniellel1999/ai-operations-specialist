# spreadsheet-cleanup-demo

LLM-assisted cleanup of a messy support-ticket spreadsheet: inconsistent
category labels, mixed date formats, missing values, and near-duplicate
rows. Pandas handles what it can reliably; a pretrained zero-shot
classifier handles the part pandas can't (free-text categorization). No
training or fine-tuning.

## Run in Colab

Upload `colab_notebook.py` to Colab (File → Upload notebook, or paste cells
in) — the `# %%` markers split it into cells automatically. Upload
`messy_support_tickets.csv` alongside it (Colab's file browser, left
sidebar) so `pd.read_csv("messy_support_tickets.csv")` can find it.
Alternatively, open it locally in VS Code / Jupyter with the Jupytext or
"Python Interactive" cell format.

## Run locally

Dependencies are declared in the [root `pyproject.toml`](../../pyproject.toml). From the repo root:

```
uv sync
```

Then open `colab_notebook.py` (see above) and skip the `!pip install` cell
— not needed locally.

## Process

```
Messy spreadsheet (messy_support_tickets.csv)
 |
 v
Load into pandas
 |
 v
Explore: missing values, inconsistent categories, near-duplicates
 |
 v
Deterministic cleaning (whitespace, dates, duplicates, priority)
 |
 v
LLM step: zero-shot classification of free-text into standard categories
 |
 v
Validate against ground truth + flag low-confidence predictions
 |
 v
Write cleaned spreadsheet (cleaned_support_tickets.xlsx)
```

## Steps

1. **Load** — a synthetic 45-row CSV styled after a real support-ticket
   export, with `category_raw` typed inconsistently by different agents.
2. **Explore** — missing values, how many distinct spellings exist for
   what should be 5 categories, near-duplicate tickets.
3. **Clean deterministically** — dedupe, trim whitespace, parse four
   different date formats into one dtype, normalize priority. No AI
   needed for any of this — it's the wrong tool for a lookup-table
   problem.
4. **Classify with an LLM** — `facebook/bart-large-mnli` zero-shot
   classification on the free-text description, against 5 categories
   (Billing, Technical, Account, Shipping, Other) described as short
   phrases rather than bare words — label wording turned out to matter a
   lot here (~64% accuracy with bare category names vs. ~75% with
   descriptive phrases, both measured against `true_category`). This is
   the part a static mapping can't cover (new/unseen spellings).
5. **Validate** — accuracy against the dataset's `true_category` ground
   truth, plus a list of low-confidence predictions to flag for human
   review rather than trusting blindly.
6. **Export** — write the cleaned, categorized data to
   `cleaned_support_tickets.xlsx`.

## Scope

Uses a pretrained classifier as-is — no training or fine-tuning. The
"AI" step is scoped narrowly to what pandas can't do (free-text
categorization); everything else is deterministic. Swap in your own
messy spreadsheet and category list to reuse this pattern.
