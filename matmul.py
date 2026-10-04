import time

import torch

CASES = [
    (32, 4000, 1),
]

for B, N, OPERATIONS in CASES:
    A = torch.randn(B, N, N)
    B_mat = torch.randn(B, N, N)

    # Warmup
    for _ in range(min(3, OPERATIONS)):
        C = torch.matmul(A, B_mat)

    start = time.perf_counter()

    for _ in range(OPERATIONS):
        C = torch.matmul(A, B_mat)

    end = time.perf_counter()

    elapsed = end - start

    gflops = 2.0 * OPERATIONS * B * N**3 / elapsed / 1e9

    print(f"B={B:2}, N={N:4}, Operations={OPERATIONS:6}")
    print(f"  Total time:   {elapsed * 1000:.2f} ms")
    print(f"  Avg operation: {elapsed * 1000 / OPERATIONS:.4f} ms")
    print(f"  GFLOP/s:      {gflops:.2f}")
    print(f"  Result:       {C[0, 0, 0].item()}")
    print()
