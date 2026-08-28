import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModel

MODEL = "BAAI/bge-m3"
TEXTS = ["hello tokenization"] * 32
WARMUP = 10
RUNS = 100
MAX_LENGTH = 8192

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Texts: {len(TEXTS)}")

t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.to(device)
model.eval()
load_time = time.perf_counter() - t0

inputs = tokenizer(
    TEXTS,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)
exit()

print(f"Model load: {load_time * 1000:.2f} ms")
print(f"Input shape: {inputs['input_ids'].shape}")

@torch.inference_mode()
def infer():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]
    return embeddings

for _ in range(WARMUP):
    infer()

if device == "cuda":
    torch.cuda.synchronize()

latencies = []

for _ in range(RUNS):
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    embeddings = infer()

    if device == "cuda":
        torch.cuda.synchronize()

    latencies.append(time.perf_counter() - start)

mean_ms = statistics.mean(latencies) * 1000
median_ms = statistics.median(latencies) * 1000
p95_ms = sorted(latencies)[int(RUNS * 0.95) - 1] * 1000
min_ms = min(latencies) * 1000
max_ms = max(latencies) * 1000

print()
print("=== BGE-M3 Benchmark ===")
print(f"Device:       {device}")
print(f"Batch size:   {len(TEXTS)}")
print(f"Runs:         {RUNS}")
print(f"Embedding:    {tuple(embeddings.shape)}")
print(f"Min:          {min_ms:.2f} ms")
print(f"Mean:         {mean_ms:.2f} ms")
print(f"Median:       {median_ms:.2f} ms")
print(f"P95:          {p95_ms:.2f} ms")
print(f"Max:          {max_ms:.2f} ms")
print(f"Throughput:   {len(TEXTS) / (mean_ms / 1000):.2f} texts/sec")

if device == "cuda":
    print(f"GPU memory:   {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")