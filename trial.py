import torch

from transformers import AutoModel, AutoTokenizer

model_path = "./bge-m3-safetensors"

model = AutoModel.from_pretrained(
    model_path,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model.eval()
texts = [
"""
This benchmark evaluates the performance of a BGE-M3 text embedding model using a moderately long natural-language input. The text is intentionally designed to contain a mixture of technical terminology, descriptive statements, and ordinary sentences so that the resulting token sequence provides a realistic workload for CPU inference. The objective is to measure the computational cost of tokenisation, transformer execution, attention mechanisms, feed-forward networks, layer normalisation, matrix multiplication, memory movement, pooling, and embedding normalisation. This benchmark can be used to compare different implementations of the same model, including SentenceTransformers with PyTorch, a native C++ implementation using standard matrix operations, an implementation accelerated with OpenBLAS, and a version using CPU SIMD instructions such as AVX2. Consistent input data, sequence length, model weights, numerical precision, and execution environment are important when comparing performance. Repeated measurements should be performed after several warm-up iterations to reduce the influence of model loading, memory allocation, CPU frequency changes, and framework initialisation. The benchmark should also record the number of generated tokens because transformer computation depends strongly on sequence length. Longer sequences increase the amount of work required by the linear projections and feed-forward layers, while self-attention introduces additional computational and memory requirements as the sequence grows. By keeping the input fixed, performance improvements can be attributed more reliably to changes in the implementation rather than differences in the workload.
"""
]
inputs = tokenizer(
    texts,
    return_tensors="pt"
)
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )
for i, hidden in enumerate(outputs.hidden_states):
    print(i, hidden.shape)
    print(hidden[0, :, :3])
    if i > 2:
        exit()