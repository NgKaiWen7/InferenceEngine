import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_ENCODER = SentenceTransformer(
    "BAAI/bge-m3",
    device=device,
)
print(TEXT_ENCODER)

texts = [
    "b"
]

import time

start = time.perf_counter()

embeddings = TEXT_ENCODER.encode(
    texts,
    normalize_embeddings=True,
    batch_size=8,
)

elapsed = time.perf_counter() - start

print(f"Encoding time: {elapsed * 1e6:.2f} us")
print(f"Encoding time: {elapsed * 1e3:.3f} ms")

print("shape:", embeddings.shape)
print(embeddings[0])