import torch

from transformers import AutoModel, AutoTokenizer

model_path = "./bge-m3-safetensors"

model = AutoModel.from_pretrained(
    model_path,
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model.eval()

inputs = tokenizer("b", return_tensors="pt")
print(inputs)
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )
print(outputs.hidden_states[2][0, :, :10])
exit()
for i, hidden in enumerate(outputs.hidden_states):
    print(i, hidden.shape)
    print(hidden[0, :, :10])