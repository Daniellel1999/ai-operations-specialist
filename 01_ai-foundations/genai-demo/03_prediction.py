"""
Step 3: Prediction (next-token logits -> probabilities)
---------------------------------------------------------

Prompt -> Tokenize -> Model -> Logits -> Softmax -> Probabilities

A language model does one core thing over and over: given the tokens
seen so far, it outputs a score ("logit") for every possible next token
in its vocabulary. Softmax turns those raw scores into probabilities
that sum to 1.

This script loads a REAL pretrained GPT-2 model, runs a real forward
pass, and prints the model's actual next-token probabilities. Nothing
here is fabricated - if you change the prompt, the numbers below will
change too.

Model note: this uses full-size "gpt2" (124M parameters) from the
Hugging Face Hub. The first run downloads ~550MB and caches it locally;
subsequent runs are instant. If your connection can't reliably fetch
that, swap MODEL_NAME for "sshleifer/tiny-gpt2" - a tiny (~2MB) real
GPT-2 checkpoint with the same architecture/tokenizer, though its
predictions will look like near-random noise since it's undertrained.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2-medium"
TOP_K = 5

prompt = "The capital of France is"

print(f"Loading pretrained model and tokenizer for '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()  # inference mode - we are not training anything

print("\nPrompt:", repr(prompt))

inputs = tokenizer(prompt, return_tensors="pt")
print("Token IDs fed to the model:", inputs["input_ids"].tolist()[0])

# Real forward pass through the actual model - no shortcuts.
with torch.no_grad():
    outputs = model(**inputs)

# outputs.logits shape: (batch, sequence_length, vocab_size)
# We only care about the logits predicting the token *after* the last one.
next_token_logits = outputs.logits[0, -1, :]
print(f"\nLogits shape for the next token: {tuple(next_token_logits.shape)}  (= vocab size)")

# Logits -> probabilities via real softmax
probabilities = torch.softmax(next_token_logits, dim=-1)

top_probs, top_ids = torch.topk(probabilities, TOP_K)

print(f"\nTop {TOP_K} most likely next tokens after {prompt!r}:")
for rank, (prob, tid) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
    token_str = tokenizer.decode([tid])
    print(f"  {rank}. {token_str!r:12} (id {tid:6})  p = {prob:.6f}")

best_id = top_ids[0].item()
print(f"\nModel's single most likely completion: {prompt!r} + {tokenizer.decode([best_id])!r}")
