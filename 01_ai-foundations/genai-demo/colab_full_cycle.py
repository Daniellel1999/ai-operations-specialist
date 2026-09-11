from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# %%
text = "The cat sat on the"
tokens = tokenizer.tokenize(text)
tokens

# %%
input_ids = tokenizer.encode(text, return_tensors="pt")
input_ids

# %%
embedding_layer = model.transformer.wte
embedding_layer.weight.shape

# %%
embeddings = embedding_layer(input_ids)
embeddings.shape

# %%
with torch.no_grad():
    outputs = model(input_ids)

logits = outputs.logits
logits.shape

# %%
next_token_logits = logits[0, -1]
next_token_logits.shape

# %%
probs = torch.softmax(next_token_logits, dim=-1)
probs.shape

# %%
top_probs, top_ids = torch.topk(probs, 5)
for p, i in zip(top_probs, top_ids):
    print(tokenizer.decode(i), p.item())

# %%
next_id = top_ids[0]
next_token = tokenizer.decode(next_id)
text + next_token
