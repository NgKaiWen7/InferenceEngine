from safetensors.torch import load_file
from transformers import AutoTokenizer
import numpy as np

weights = load_file("bge-m3-safetensors/model.safetensors")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

weight = weights["embeddings.word_embeddings.weight"]

text = "h"
ids = tokenizer.encode(text, add_special_tokens=False)

print("IDs:", ids)
print("Tokens:", tokenizer.convert_ids_to_tokens(ids))

for token_id in ids:
    embedding = weight[token_id].numpy()

    print(f"\ntoken_id = {token_id}")
    print("shape:", embedding.shape)
    print("first 10:", embedding[:10])
    print("min:", embedding.min())
    print("max:", embedding.max())
    print("mean:", embedding.mean())
    print("L2:", np.linalg.norm(embedding))