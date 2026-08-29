import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

print("vocab_size:", tokenizer.vocab_size)
print("len(tokenizer):", len(tokenizer))

for i in range(0, 10):
    print(i, repr(tokenizer.convert_ids_to_tokens(i)))
exit()
model_name = "BAAI/bge-m3"

weights = load_file("bge-m3-safetensors/model.safetensors")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "b"
inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("Input IDs:", input_ids.tolist())
print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))
print("Attention mask:", attention_mask.tolist())

position_ids = model.embeddings.create_position_ids_from_input_ids(
    input_ids,
    model.config.pad_token_id
)

word_weight = weights["embeddings.word_embeddings.weight"]
position_weight = weights["embeddings.position_embeddings.weight"]

word = word_weight[input_ids]
position = position_weight[position_ids]

hidden = word + position

gamma = weights["embeddings.LayerNorm.weight"]
beta = weights["embeddings.LayerNorm.bias"]

mean = hidden.mean(dim=-1, keepdim=True)
var = hidden.var(dim=-1, unbiased=False, keepdim=True)

hidden_norm = (hidden - mean) / torch.sqrt(var + 1e-5)
hidden_norm = hidden_norm * gamma + beta

print("IDs:", input_ids.tolist())
print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))
print("Position IDs:", position_ids.tolist())

for i in range(input_ids.shape[1]):
    print(f"\n===== Token {i} =====")
    print("token_id:", input_ids[0, i].item())
    print("position_id:", position_ids[0, i].item())

    print("\nWord embedding:")
    print(word[0, i, :10].numpy())

    print("\nPosition embedding:")
    print(position[0, i, :10].numpy())

    print("\nWord + position:")
    print(hidden[0, i, :10].numpy())

    print("\nLayerNorm:")
    print(hidden_norm[0, i, :10].numpy())