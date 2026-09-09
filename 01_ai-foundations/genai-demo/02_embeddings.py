"""
Step 2: Embeddings
-------------------

Token ID -> Embedding lookup -> Vector

A token ID is just an arbitrary integer - it has no notion of "meaning".
To let a neural network reason about tokens, each ID is used to look up
a row in an embedding table: a matrix of shape (vocab_size, embedding_dim)
where every row is a learned vector for one token.

This script builds a REAL PyTorch nn.Embedding layer and performs a real
lookup using token IDs produced by the actual GPT-2 tokenizer. The vectors
printed below come directly from that lookup - they are not hardcoded.
"""

import torch
from torch import nn
from transformers import AutoTokenizer

MODEL_NAME = "gpt2"
EMBEDDING_DIM = 8  # kept small so the printed vectors stay readable

text = "The cat sat on the mat."

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Text:", repr(text))
print("Tokens:", tokens)
print("Token IDs:", token_ids)

# A real embedding layer: one learnable vector per vocabulary entry.
# (Untrained/random-initialized here - the point is to demonstrate the
# *mechanism* of embedding lookup, not to reuse GPT-2's own trained weights.)
torch.manual_seed(0)  # only affects the random init, not the "results" we report
embedding_layer = nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=EMBEDDING_DIM)

ids_tensor = torch.tensor(token_ids)

# Token ID -> embedding lookup -> vector (a real forward pass through nn.Embedding)
with torch.no_grad():
    vectors = embedding_layer(ids_tensor)

print(f"\nEmbedding table shape: {tuple(embedding_layer.weight.shape)}  (vocab_size x embedding_dim)")
print(f"Looked-up vectors shape: {tuple(vectors.shape)}  (num_tokens x embedding_dim)")

print("\nToken -> ID -> embedding vector:")
for tok, tid, vec in zip(tokens, token_ids, vectors):
    values = ", ".join(f"{v:+.3f}" for v in vec.tolist())
    print(f"  {tok!r:12} -> id {tid:6} -> [{values}]")

print(
    "\nEach row above is a real vector pulled straight out of the embedding "
    "table (embedding_layer.weight) at the row index given by the token ID."
)
