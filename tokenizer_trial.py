import torch
from transformers import AutoModelForCausalLM

model_path = "./models/qwen3-4B"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Same token ids as C++
token_ids = torch.tensor([[1, 2, 4]])

with torch.no_grad():
    embeddings = model.model.embed_tokens(token_ids)

print(embeddings.shape)
# [1, 3, hidden_size]

for i, token_id in enumerate(token_ids[0]):
    print("Token:", token_id.item())

    print(
        embeddings[0, i, :10]
    )