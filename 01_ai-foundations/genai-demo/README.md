# genai-demo

A tiny, runnable demonstration of the basic process behind generative AI / LLMs.
Every number printed by these scripts comes from a real Hugging Face
tokenizer/model or a real PyTorch operation — nothing is hardcoded or faked.

## Setup

```bash
cd 01_ai-foundations/genai-demo
uv sync
```

## Run, in order

```bash
uv run python 01_tokenization.py
uv run python 02_embeddings.py
uv run python 03_prediction.py
uv run python 04_generation.py
uv run python 05_real_llm.py
```

The first run of each model-based script will download the small `gpt2`
checkpoint from Hugging Face (a few hundred MB, cached afterwards).

## The big picture

```
Text
 |
 v
Tokenization
 |
 v
Token IDs
 |
 v
Embedding
 |
 v
Neural network / Transformer
 |
 v
Logits
 |
 v
Probabilities
 |
 v
Next-token selection
 |
 v
Repeat
 |
 v
Generated text
```

- **01_tokenization.py** — text to tokens to token IDs, and back again.
- **02_embeddings.py** — token ID to vector, via a real `nn.Embedding` lookup.
- **03_prediction.py** — a real pretrained GPT-2 scores every possible next
  token; softmax turns those scores into probabilities.
- **04_generation.py** — the autoregressive loop, written out explicitly:
  predict, sample, append, repeat.
- **05_real_llm.py** — the same idea via Hugging Face's high-level
  `pipeline("text-generation", ...)` API. This is inference on a pretrained
  model — no training happens anywhere in this project.

## Scope

This is an introductory demo, not an ML course. It intentionally does not
implement a Transformer from scratch, attention math, backpropagation,
gradient descent, training, or fine-tuning.
