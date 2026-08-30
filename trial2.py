from sentence_transformers import SentenceTransformer

TEXT_ENCODER = SentenceTransformer(
    "BAAI/bge-m3"
)

texts = [
        "This is a longer sample sentence for benchmarking the BGE-M3 inference engine. "
        "The purpose is to evaluate the performance of tokenisation, embedding generation, "
        "transformer computation, attention, feed forward layers, layer normalisation, "
        "and memory access patterns under a more realistic input sequence. "
        "We want to compare the performance of the baseline C++ implementation against "
        "an implementation optimised with OpenBLAS and CPU SIMD instructions. "
        "The input should contain enough tokens to make computational differences measurable "
        "while remaining representative of typical semantic embedding workloads."
]

import time

start = time.perf_counter()

embeddings = TEXT_ENCODER.encode(
    texts,
    normalize_embeddings=True
)

elapsed = time.perf_counter() - start

print(f"Encoding time: {elapsed * 1e6:.2f} us")
print(f"Encoding time: {elapsed * 1e3:.3f} ms")

print("shape:", embeddings.shape)
print(embeddings[0])