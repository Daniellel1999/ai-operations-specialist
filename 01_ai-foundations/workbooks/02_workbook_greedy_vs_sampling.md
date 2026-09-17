# In-Class Workbook: Greedy vs. Sampling (10 minutes)

Paste this into a Google Colab cell and run it. It rebuilds the same
`probs` distribution from `genai-demo/colab_full_cycle.py`, then always
picks the single most likely next token — this is called **greedy
decoding**.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

text = "The cat sat on the"
input_ids = tokenizer.encode(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids)

next_token_logits = outputs.logits[0, -1]
probs = torch.softmax(next_token_logits, dim=-1)

# Greedy: always the single most likely token
next_id = torch.topk(probs, 1).indices[0]
print(text + tokenizer.decode(next_id))
```

## Task 1: Sample instead of picking the top token

Replace the last two lines above with a **sampled** token instead of the
greedy top choice. `torch.multinomial(probs, num_samples=1)` draws one
token ID, weighted by its probability.

```python
# TODO: replace `next_id` with a sampled token
next_id = ___________________________
print(text + tokenizer.decode(next_id))
```

Run it 5 times in a row. Did you get the same word every time? Did any
surprise you?

<details>
<summary>💡 Answer</summary>

```python
next_id = torch.multinomial(probs, num_samples=1)[0]
```

</details>

## Task 2: Generate 10 tokens with sampling

Wrap the pipeline in a loop, feeding each new token back in as input.

```python
generated = input_ids

for _ in range(10):
    with torch.no_grad():
        outputs = model(generated)
    next_token_logits = outputs.logits[0, -1]
    probs = torch.softmax(next_token_logits, dim=-1)

    # TODO 1: sample the next token id from probs
    next_id = ___________________________
    # TODO 2: append it to `generated`
    generated = ___________________________

print(tokenizer.decode(generated[0]))
```

<details>
<summary>💡 Answer</summary>

```python
next_id = torch.multinomial(probs, num_samples=1)
generated = torch.cat([generated, next_id.unsqueeze(0)], dim=-1)
```

`next_id` needs shape `[1, 1]` to concatenate along the sequence
dimension, hence `unsqueeze(0)`.

</details>

## Discussion (2 minutes)

Greedy decoding is deterministic; sampling is stochastic. Which would you
want for a chatbot? For a customer-facing summarizer? Why might
inconsistency be a problem in production?
