import time
import numpy as np
import torch

M, K, N = 512, 1024, 4096
WARMUP = 5
ITERATIONS = 20

A = np.random.default_rng(42).uniform(-1, 1, (M, K)).astype(np.float32)
B = np.random.default_rng(43).uniform(-1, 1, (K, N)).astype(np.float32)

A_t = torch.from_numpy(A)
B_t = torch.from_numpy(B)

ops = 2.0 * M * N * K

def benchmark(fn):
    for _ in range(WARMUP):
        fn()

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        C = fn()
    elapsed = (time.perf_counter() - start) / ITERATIONS

    gflops = ops / elapsed / 1e9
    return elapsed * 1000, gflops, C

def numpy_benchmark():
    for threads in [1, 2, 4, 8, 12, 16, 20]:
        import os
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

        for _ in range(WARMUP):
            C = A @ B

        start = time.perf_counter()
        for _ in range(ITERATIONS):
            C = A @ B
        elapsed = (time.perf_counter() - start) / ITERATIONS

        gflops = ops / elapsed / 1e9
        print(f"NumPy {threads:2d} threads: {elapsed*1000:8.3f} ms  {gflops:8.2f} GFLOP/s")

print(f"M={M}, K={K}, N={N}\n")

torch.set_num_threads(1)

ms, gflops, C = benchmark(lambda: torch.mm(A_t, B_t))
print("PyTorch CPU - 1 thread:")
print(f"  {ms:.3f} ms")
print(f"  {gflops:.2f} GFLOP/s")
print(f"  C[0,0] = {C[0,0].item():.6f}\n")
torch.set_num_threads(8)

ms, gflops, C = benchmark(lambda: torch.mm(A_t, B_t))
print("PyTorch CPU - 8 threads:")
print(f"  {ms:.3f} ms")
print(f"  {gflops:.2f} GFLOP/s")
print(f"  C[0,0] = {C[0,0].item():.6f}\n")

for threads in [1, 2, 4, 8, 12, 16, 20]:
    torch.set_num_threads(threads)
    ms, gflops, _ = benchmark(lambda: torch.mm(A_t, B_t))
    print(f"PyTorch {threads:2d} threads: {ms:8.3f} ms  {gflops:8.2f} GFLOP/s")
numpy_benchmark()
