from sentence_transformers import SentenceTransformer
import time

TEXT_ENCODER = SentenceTransformer("BAAI/bge-m3", local_files_only=True)

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

start = time.perf_counter()

features = TEXT_ENCODER.tokenize(texts)

tokenize_time = time.perf_counter() - start

start = time.perf_counter()

embeddings = TEXT_ENCODER.encode(
    texts,
    normalize_embeddings=True
)

encode_time = time.perf_counter() - start

print(f"Tokenization: {tokenize_time * 1000:.3f} ms")
print(f"Embedding:    {encode_time * 1000:.3f} ms")
print(f"Total:        {(tokenize_time + encode_time) * 1000:.3f} ms")