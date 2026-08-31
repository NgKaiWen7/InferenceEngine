text = (
    "This is a stress test for the BGE-M3 inference engine. "
    "The purpose is to generate a sufficiently long input sequence "
    "to evaluate tokenisation, embedding generation, transformer "
    "computation, attention, feed-forward layers, layer normalisation, "
    "memory access patterns, SIMD operations, and OpenBLAS performance. "
)

text = (text * (10000 // len(text) + 1))[:8000]

with open("stress_test.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Generated {len(text)} characters.")