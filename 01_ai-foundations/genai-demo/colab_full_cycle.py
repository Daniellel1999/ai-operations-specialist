# Load GPT-2 and its tokenizer, set to eval mode (no training/dropout)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# %%
# Step 1: split text into subword tokens
text = "The cat sat on the"
tokens = tokenizer.tokenize(text)
tokens

# %%
# Step 2: convert tokens to their integer IDs (model input)
input_ids = tokenizer.encode(text, return_tensors="pt")
input_ids

# %%
# Step 3: look at the embedding table that maps token IDs to vectors
embedding_layer = model.transformer.wte
embedding_layer.weight.shape

# %%
# Step 4: embed the input tokens into vectors
embeddings = embedding_layer(input_ids)
embeddings.shape

# %%
# Step 5: run the forward pass through the full model to get logits per token
with torch.no_grad():
    outputs = model(input_ids)

logits = outputs.logits
logits.shape

# %%
# Step 6: take the logits for the last token, which predict the next token
next_token_logits = logits[0, -1]
next_token_logits.shape

# %%
# Step 7: convert logits into a probability distribution over the vocabulary
probs = torch.softmax(next_token_logits, dim=-1)
probs.shape

# %%
# Step 8: inspect the 5 most likely next tokens and their probabilities
top_probs, top_ids = torch.topk(probs, 5)
for p, i in zip(top_probs, top_ids):
    print(tokenizer.decode(i), p.item())

# %%
# Step 9: greedily pick the top candidate and append it to the original text
next_id = top_ids[0]
next_token = tokenizer.decode(next_id)
text + next_token
