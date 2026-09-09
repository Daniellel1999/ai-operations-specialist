"""
Step 1: Tokenization
---------------------

Text
 -> Tokenizer
 -> Tokens
 -> Token IDs

Every LLM starts here: raw text has to be broken into a fixed vocabulary
of "tokens" (roughly sub-words), each of which has a unique integer ID.
That ID is the only thing the neural network actually understands.

This script uses the REAL GPT-2 tokenizer from Hugging Face - nothing
below is hardcoded or faked.
"""

from transformers import AutoTokenizer

MODEL_NAME = "gpt2"

text = "The cat sat on the mat."

print(f"Loading tokenizer for '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nInput text:")
print(f"  {text!r}")

# Text -> Tokens (real tokenizer output, not hardcoded)
tokens = tokenizer.tokenize(text)
print("\nTokens (from tokenizer.tokenize):")
print(f"  {tokens}")

# Text -> Token IDs (real tokenizer output)
token_ids = tokenizer.encode(text)
print("\nToken IDs (from tokenizer.encode):")
print(f"  {token_ids}")

# Show the token <-> id pairing explicitly
print("\nToken -> ID mapping:")
for tok, tid in zip(tokens, token_ids):
    print(f"  {tok!r:12} -> {tid}")

# Reverse operation: IDs -> text, proving the mapping is invertible
decoded_text = tokenizer.decode(token_ids)
print("\nDecoded back to text (from tokenizer.decode):")
print(f"  {decoded_text!r}")

print(f"\nVocabulary size of this tokenizer: {tokenizer.vocab_size}")
