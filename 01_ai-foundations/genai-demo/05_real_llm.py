"""
Step 5: The High-Level API
-----------------------------

Everything in 01-04 (tokenize -> embed -> predict logits -> sample ->
repeat) is exactly what happens inside Hugging Face's `pipeline`
helper. This script shows the same result produced with just a few
lines, using the high-level `text-generation` API.

Important: we are using a PRETRAINED model here. We are performing
INFERENCE (asking the model to produce output), not training. No
weights are being updated - the model checkpoint is used exactly as
downloaded.

Model note: this uses full-size "gpt2" (124M parameters) from the
Hugging Face Hub. The first run downloads ~550MB and caches it locally;
subsequent runs are instant. If your connection can't reliably fetch
that, swap MODEL_NAME for "sshleifer/tiny-gpt2" - a tiny (~2MB) real
GPT-2 checkpoint with the same architecture/tokenizer, though its
output will look like near-random noise since it's undertrained.
"""

from transformers import pipeline, set_seed
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # keep the console focused on the demo, not internal warnings

MODEL_NAME = "gpt2"

print(f"Loading pretrained model '{MODEL_NAME}' via the text-generation pipeline...")
print("(This downloads/uses an already-trained model - no training happens here.)")

generator = pipeline("text-generation", model=MODEL_NAME)

set_seed(42)  # reproducibility for the sampling step only

prompt = "In the future, artificial intelligence will"

print("\nPrompt:", repr(prompt))
print("Running inference (model.generate under the hood)...\n")

results = generator(
    prompt,
    max_new_tokens=30,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.8,
)

generated_text = results[0]["generated_text"]
print("Generated text (real model output):")
print(f"  {generated_text!r}")

print(
    "\nUnder the hood, `pipeline(...)` just automated the exact loop from "
    "04_generation.py: tokenize the prompt, run the model, turn logits into "
    "probabilities, sample a token, append it, and repeat until max_new_tokens "
    "is reached."
)
