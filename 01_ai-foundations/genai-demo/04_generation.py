"""
Step 4: Autoregressive Generation
------------------------------------

Prompt
 -> Tokenize
 -> Model
 -> Next-token logits
 -> Probabilities
 -> Select next token
 -> Append token
 -> Run model again
 -> Repeat

This is the core loop behind every text-generating LLM: predict one
token, glue it onto the sequence, and feed the whole thing back in to
predict the next one. We implement that loop explicitly (instead of
calling a high-level `.generate()` helper) so every step is visible.

The model is a REAL pretrained GPT-2. Every generated token is an
actual sampling decision made from the model's real output
probabilities at that step - nothing is scripted in advance.

Model note: this uses full-size "gpt2" (124M parameters) from the
Hugging Face Hub. The first run downloads ~550MB and caches it locally;
subsequent runs are instant. If your connection can't reliably fetch
that, swap MODEL_NAME for "sshleifer/tiny-gpt2" - a tiny (~2MB) real
GPT-2 checkpoint with the same architecture/tokenizer, though its
output will look like near-random noise since it's undertrained.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
NUM_TOKENS_TO_GENERATE = 12
TEMPERATURE = 0.8  # >0 softens/sharpens the probability distribution before sampling

prompt = "Once upon a time,"

print(f"Loading pretrained model and tokenizer for '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

torch.manual_seed(42)  # only makes the sampling reproducible run-to-run; the model itself is untouched

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

print("\nPrompt:", repr(prompt))
print("Generating token by token:\n")

generated_ids = input_ids
for step in range(1, NUM_TOKENS_TO_GENERATE + 1):
    # Run the real model on everything generated so far.
    with torch.no_grad():
        outputs = model(generated_ids)

    # Logits for the very next token.
    next_token_logits = outputs.logits[0, -1, :]

    # Probabilities via real softmax (temperature reshapes the distribution).
    probabilities = torch.softmax(next_token_logits / TEMPERATURE, dim=-1)

    # Real sampling operation - draw one token id from the probability distribution.
    next_token_id = torch.multinomial(probabilities, num_samples=1)

    chosen_prob = probabilities[next_token_id].item()
    chosen_token_str = tokenizer.decode(next_token_id)

    print(
        f"  step {step:2}: sampled token {chosen_token_str!r:12} "
        f"(id {next_token_id.item():6}, p = {chosen_prob:.6f})"
    )

    # Append the new token and feed the extended sequence back in.
    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

full_text = tokenizer.decode(generated_ids[0])
print("\nFinal generated text:")
print(f"  {full_text!r}")
