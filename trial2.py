import torch
import numpy as np
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

text = "b"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    # --------------------------------------------------
    # Embedding
    # --------------------------------------------------

    embeddings = model.embeddings(
        input_ids=inputs["input_ids"]
    )

    # --------------------------------------------------
    # Layer 0
    # --------------------------------------------------

    layer = model.encoder.layer[0]

    self_attn = layer.attention.self
    attention_output = layer.attention.output

    # Q / K / V
    query = self_attn.query(embeddings)
    key = self_attn.key(embeddings)
    value = self_attn.value(embeddings)

    batch_size, sequence_length, _ = query.shape

    num_heads = self_attn.num_attention_heads
    head_dim = self_attn.attention_head_size

    # --------------------------------------------------
    # Split heads
    # --------------------------------------------------

    query = query.view(
        batch_size,
        sequence_length,
        num_heads,
        head_dim
    ).transpose(1, 2)

    key = key.view(
        batch_size,
        sequence_length,
        num_heads,
        head_dim
    ).transpose(1, 2)

    value = value.view(
        batch_size,
        sequence_length,
        num_heads,
        head_dim
    ).transpose(1, 2)

    # --------------------------------------------------
    # Q × K^T
    # --------------------------------------------------

    scores = torch.matmul(
        query,
        key.transpose(-1, -2)
    )

    # --------------------------------------------------
    # Scale
    # --------------------------------------------------

    scores = scores / np.sqrt(head_dim)

    # --------------------------------------------------
    # Softmax
    # --------------------------------------------------

    probabilities = torch.softmax(
        scores,
        dim=-1
    )

    # --------------------------------------------------
    # Attention × V
    # --------------------------------------------------

    context = torch.matmul(
        probabilities,
        value
    )

    # --------------------------------------------------
    # Concatenate heads
    # --------------------------------------------------

    context = context.transpose(
        1, 2
    ).contiguous()

    context = context.view(
        batch_size,
        sequence_length,
        1024
    )

    # --------------------------------------------------
    # attention.output.dense
    # --------------------------------------------------

    attention_dense = attention_output.dense(
        context
    )

    # --------------------------------------------------
    # Residual
    # --------------------------------------------------

    residual = attention_dense + embeddings

    # --------------------------------------------------
    # attention.output.LayerNorm
    # --------------------------------------------------

    attention_layer_output = attention_output.LayerNorm(
        residual
    )

    # --------------------------------------------------
    # Intermediate dense
    # 1024 -> 4096
    # --------------------------------------------------

    intermediate = layer.intermediate.dense(
        attention_layer_output
    )

    # --------------------------------------------------
    # GELU
    # --------------------------------------------------

    intermediate = layer.intermediate.intermediate_act_fn(
        intermediate
    )

    # --------------------------------------------------
    # Output dense
    # 4096 -> 1024
    # --------------------------------------------------

    output_dense = layer.output.dense(
        intermediate
    )

    # --------------------------------------------------
    # Residual
    # --------------------------------------------------

    output_residual = output_dense + attention_layer_output

    # --------------------------------------------------
    # Final LayerNorm
    # --------------------------------------------------

    layer_output = layer.output.LayerNorm(
        output_residual
    )


# ------------------------------------------------------
# Print first 10 values of each sequence
# ------------------------------------------------------

def print_first_10(name, tensor):
    print(f"\n{name}")

    tensor = tensor[0]

    for i in range(tensor.shape[0]):
        print(
            i,
            tensor[i, :10].tolist()
        )


print_first_10("Embeddings", embeddings)

print_first_10("Query", query.transpose(1, 2).reshape(
    batch_size, sequence_length, 1024
))

print_first_10("Key", key.transpose(1, 2).reshape(
    batch_size, sequence_length, 1024
))

print_first_10("Value", value.transpose(1, 2).reshape(
    batch_size, sequence_length, 1024
))

print_first_10("Context", context)

print_first_10("Attention Dense", attention_dense)

print_first_10("Attention Residual", residual)

print_first_10("Attention LayerNorm", attention_layer_output)

print_first_10("Intermediate", intermediate)

print_first_10("Output Dense", output_dense)

print_first_10("Output Residual", output_residual)

print_first_10("Layer 0 Output", layer_output)