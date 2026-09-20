# In-Class Workbook: Tokenization Scavenger Hunt (10 minutes)

Paste this into a Google Colab cell and run it. No model forward pass
needed — just the tokenizer.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def show(text):
    print(text, "->", tokenizer.tokenize(text))

show("The cat sat on the mat")
```

## Task 1: Predict before you run

For each input below, **write down your guess** for how many tokens it
splits into, then run it and check.

```python
show("supercalifragilisticexpialidocious")
show("2026")
show("🐍🔥")
show("antidisestablishmentarianism")
show("don't")
show("misspeling")
```

| Input | Your guess | Actual tokens |
|---|---|---|
| supercalifragilisticexpialidocious | 13 | 11 |
| 2026 | 2 | 2 |
| 🐍🔥 | 4 | 6 |
| antidisestablishmentarianism | 10 | 5 |
| don't | 3 | 2 |
| misspeling | 3 | 3 |

<details>
<summary>💡 What's actually happening</summary>

GPT-2 uses **byte-pair encoding (BPE)**: common whole words get one token,
rare or long words get chopped into subword pieces, and unfamiliar
symbols (emoji, rare unicode) often split into multiple small tokens or
raw bytes. Numbers are frequently split in non-obvious ways (e.g. "2026"
might not be a single token). This is why token counts — and API
pricing, which is per-token — don't map cleanly to word counts.

</details>

## Task 2: Find a word that splits into 3+ tokens

Try your own words below (names, made-up words, technical jargon) and
find one that splits into **3 or more** tokens.

```python
show("Elnekave")
```

Write the word you found and its token split here: Elnekave -> ['El', 'ne', 'k', 'ave']

## Task 3: Same word, different case/spacing

Run all three and compare — are the token IDs the same?

```python
show("hello")
show("Hello")
show(" hello")
```

<details>
<summary>💡 Answer</summary>

All three tokenize differently. GPT-2's tokenizer is **case-sensitive**
and treats a leading space as part of the token (it was trained on raw
web text, where most words are preceded by a space). This is why
"hello", "Hello", and " hello" are three distinct tokens/token-sequences
to the model, not the same word.

</details>

## Discussion (2 minutes)

If tokens (not words) are what you pay for and what count toward a
model's context limit, why might that matter when writing prompts in
languages other than English, or when working with code/numbers?
