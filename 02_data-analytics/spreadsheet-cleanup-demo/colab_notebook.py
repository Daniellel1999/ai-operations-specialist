# %% [markdown]
# # LLM-Powered Spreadsheet Cleanup & Analysis
#
# A messy support-ticket spreadsheet — inconsistent category labels, mixed
# date formats, missing values, near-duplicate rows — cleaned up with
# pandas, then categorized with an LLM (zero-shot classification) where
# pandas alone can't do the job. We validate the LLM's output against
# ground-truth labels before trusting it, the same way session 2 validated
# the sentiment model.

# %%
!pip install -q pandas transformers openpyxl

# %%
!uv sync

# %%
import pandas as pd
from transformers import pipeline

# %%
dir(pd)

# %% [markdown]
# ## 1. Load the messy spreadsheet
#
# `messy_support_tickets.csv` is a small synthetic dataset styled after a
# real support-ticket export: the `category_raw` column was typed by
# different agents over time ("billing", "BILLING", "Bill Issue", "billng"),
# dates are in four different formats, and a few rows are missing a
# category entirely. `true_category` is included only so we can *measure*
# how well our cleanup does — a real spreadsheet wouldn't have it.

# %%
# Upload messy_support_tickets.csv from your local machine (Colab only)
from google.colab import files
uploaded = files.upload()

# %%
df = pd.read_csv("messy_support_tickets.csv")
df.head(10)

# %% [markdown]
# ## 2. Explore the mess
#
# Quantify the problem before fixing it — this is the same EDA reflex as
# session 2, just applied to data quality instead of class balance.

# %%
print(f"{len(df)} rows, {df.shape[1]} columns")
df.isna().sum()

# %%
# How many distinct spellings exist for what should be ~5 categories?
print(f"{df['category_raw'].nunique(dropna=True)} distinct category_raw values for what should be 5 categories")
df["category_raw"].value_counts(dropna=False)

# %%
# Near-duplicate tickets: same customer + description, different ticket_id
dupes = df[df.duplicated(subset=["customer", "description"], keep=False)]
dupes[["ticket_id", "customer", "description"]]

# %% [markdown]
# ## 3. Deterministic cleaning first
#
# Whitespace, casing, duplicates, and date formats are all things pandas
# handles reliably and cheaply — no need to call an LLM for any of this.
# Save the AI step for the part pandas genuinely can't do.

# %%
clean = df.copy()

# Drop exact duplicate tickets (same customer + description)
clean = clean.drop_duplicates(subset=["customer", "description"], keep="first")

# Normalize whitespace/casing on text fields
clean["customer"] = clean["customer"].str.strip()
clean["description"] = clean["description"].str.strip()

# Parse the four different date formats into one consistent dtype
clean["date"] = pd.to_datetime(clean["date_raw"], format="mixed", dayfirst=False)
clean = clean.drop(columns=["date_raw"])

# Normalize priority to a fixed set of values
clean["priority"] = clean["priority_raw"].str.strip().str.lower().map(
    {"high": "High", "medium": "Medium", "med": "Medium", "low": "Low"}
)
clean = clean.drop(columns=["priority_raw"])

print(f"{len(df)} rows -> {len(clean)} rows after removing duplicates")
clean[["ticket_id", "customer", "date", "priority"]].head(10)

# %% [markdown]
# ## 4. LLM step: normalize category_raw with zero-shot classification
#
# This is the part a lookup table can't fully solve — new/unseen spellings
# ("billng", "Bill Issue") would need endless manual mapping rules. Instead
# we classify the *description* text directly against a fixed label set
# using `facebook/bart-large-mnli`, a pretrained zero-shot classifier: no
# training, no fine-tuning, same "pretrained model as-is" pattern as
# session 2's sentiment pipeline.
#
# Label wording matters more than you'd expect: bare category names like
# `"Billing"` scored only ~64% accuracy here, with the model systematically
# confusing billing complaints ("charged twice", "invoice") for "Account".
# Phrasing each label as a short descriptive phrase gets it to ~75% — a
# good reminder that zero-shot classification is sensitive to how you
# describe the categories, not just which model you pick.

# %%
CATEGORIES = {
    "Billing": "billing, invoices, charges, or payments",
    "Technical": "app bugs, crashes, or technical problems",
    "Account": "account login, access, or authentication",
    "Shipping": "shipping, delivery, or order fulfillment",
    "Other": "general feedback or other topics",
}
candidate_labels = list(CATEGORIES.values())
label_to_category = {v: k for k, v in CATEGORIES.items()}

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# %%
descriptions = clean["description"].tolist()
results = classifier(descriptions, candidate_labels=candidate_labels, batch_size=8)

clean["predicted_category"] = [label_to_category[r["labels"][0]] for r in results]
clean["predicted_score"] = [r["scores"][0] for r in results]
clean[["description", "category_raw", "predicted_category", "predicted_score"]].head(10)

# %% [markdown]
# ## 5. Validate before trusting it
#
# Same discipline as session 2: don't just eyeball a few rows, measure
# agreement against ground truth, and look at where it disagrees.

# %%
accuracy = (clean["predicted_category"] == clean["true_category"]).mean()
print(f"LLM category accuracy vs ground truth: {accuracy:.2%}")

# %%
low_confidence = clean[clean["predicted_score"] < 0.6]
print(f"{len(low_confidence)} predictions below 0.6 confidence — flag these for human review")
low_confidence[["description", "predicted_category", "predicted_score"]]

# %%
mistakes = clean[clean["predicted_category"] != clean["true_category"]]
print(f"{len(mistakes)} disagreements out of {len(clean)} rows")
mistakes[["description", "true_category", "predicted_category", "predicted_score"]]

# %% [markdown]
# ## 6. Write the cleaned spreadsheet back out
#
# The output an ops person would actually use: one clean sheet with a
# standardized category, plus a `predicted_score` column so low-confidence
# rows are easy to filter and hand-check rather than blindly trusted.

# %%
output = clean[
    ["ticket_id", "customer", "description", "date", "priority",
     "predicted_category", "predicted_score"]
].rename(columns={"predicted_category": "category"})

output.to_excel("cleaned_support_tickets.xlsx", index=False)
output.head(10)

# %%
