# %% [markdown]
# # Sentiment Analysis on a Hugging Face Dataset
#
# Load a real dataset, explore it with pandas, and run sentiment analysis
# with a pretrained Hugging Face model. No training, no fine-tuning —
# this is data analytics + inference on an existing model.

# %%
!pip install -q datasets transformers pandas matplotlib seaborn

# %%
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# %% [markdown]
# ## Optional: Hugging Face login
#
# Not required for IMDB (it's public) — only needed for gated/private
# datasets or models, or to avoid anonymous rate limits.
#
# In Colab: click the key icon in the left sidebar, add a secret named
# `HF_TOKEN` with your token from https://huggingface.co/settings/tokens,
# then run this cell.

# %%
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get("HF_TOKEN"))

# %% [markdown]
# ## 1. Load the dataset
#
# IMDB movie reviews: 50k reviews labeled positive/negative.
# We only use the label as ground truth to check our model later —
# the sentiment pipeline never sees it.

# %%
dataset = load_dataset("stanfordnlp/imdb")
dataset

# %%
# Work with a manageable sample of the test split
df = dataset["test"].shuffle(seed=42).select(range(200)).to_pandas()
df["label_name"] = df["label"].map({0: "negative", 1: "positive"})
df.head()

# %% [markdown]
# ## 2. Explore the data

# %%
df["label_name"].value_counts()

# %%
df["review_length"] = df["text"].str.len()
df["review_length"].describe()

# %%
sns.histplot(df["review_length"], bins=30)
plt.title("Review length distribution")
plt.xlabel("Characters")
plt.show()

# %%
df.isna().sum()

# %% [markdown]
# ## 3. Light cleaning
#
# Transformer models handle raw text well — we just strip the
# leftover HTML line breaks IMDB reviews are known to contain.

# %%
df["clean_text"] = df["text"].str.replace("<br />", " ", regex=False)
df["clean_text"].iloc[0][:300]

# %% [markdown]
# ## 4. Sentiment analysis with a pretrained pipeline
#
# `distilbert-base-uncased-finetuned-sst-2-english` — a small, fast model
# fine-tuned for binary sentiment. Truncated to its 512-token limit.

# %%
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
)

# %%
sample_texts = df["clean_text"].tolist()
results = sentiment_pipeline(sample_texts, batch_size=16)

df["predicted_label"] = [r["label"].lower() for r in results]
df["predicted_score"] = [r["score"] for r in results]
df[["label_name", "predicted_label", "predicted_score"]].head(10)

# %% [markdown]
# ## 5. Evaluate against ground truth

# %%
accuracy = (df["label_name"] == df["predicted_label"]).mean()
print(f"Accuracy vs IMDB labels: {accuracy:.2%}")

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(df["label_name"], df["predicted_label"], labels=["negative", "positive"])
ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot()
plt.title("Predicted vs actual sentiment")
plt.show()

# %% [markdown]
# ## 6. A look at the disagreements
#
# Where the model got it wrong is often the most interesting part —
# sarcasm, mixed reviews, or reviews that praise the plot but pan the acting.

# %%
mistakes = df[df["label_name"] != df["predicted_label"]]
print(f"{len(mistakes)} disagreements out of {len(df)} reviews")
mistakes[["clean_text", "label_name", "predicted_label", "predicted_score"]].head(5)

# %% [markdown]
# ## 7. Wrap-up
#
# - Dataset: IMDB reviews (200-row sample of the test split)
# - Model: DistilBERT fine-tuned for SST-2 sentiment
# - Result: prints the accuracy above, plus a confusion matrix and a
#   handful of misclassified examples to eyeball
#
# Swap `dataset = load_dataset("stanfordnlp/imdb")` for any other Hugging Face
# text dataset to reuse this notebook end to end.
