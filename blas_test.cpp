#include <cblas.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

void matmul_naive(const float *A, const float *B, float *C, int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void matmul_blas(const float *A, const float *B, float *C, int M, int K, int N)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

template <typename Func>
double benchmark(Func func, int iterations)
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
        func();

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end - start).count() / iterations;
}

int main()
{
    const int M = 512;
    const int K = 1024;
    const int N = 4096;

    const int warmup = 3;
    const int iterations = 20;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &x : A)
        x = dist(gen);
    for (auto &x : B)
        x = dist(gen);

    for (int i = 0; i < warmup; ++i)
        matmul_naive(A.data(), B.data(), C.data(), M, K, N);

    double naive = benchmark([&]
                             { matmul_naive(A.data(), B.data(), C.data(), M, K, N); }, iterations);

    for (int i = 0; i < warmup; ++i)
        matmul_blas(A.data(), B.data(), C.data(), M, K, N);

    double blas = benchmark([&]
                            { matmul_blas(A.data(), B.data(), C.data(), M, K, N); }, iterations);

    double operations = 2.0 * M * N * K;

    std::cout << "M = " << M
              << ", K = " << K
              << ", N = " << N << "\n\n";

    std::cout << "Naive C++:\n";
    std::cout << "  " << naive * 1000.0 << " ms\n";
    std::cout << "  " << operations / naive / 1e9 << " GFLOP/s\n\n";

    std::cout << "OpenBLAS:\n";
    std::cout << "  " << blas * 1000.0 << " ms\n";
    std::cout << "  " << operations / blas / 1e9 << " GFLOP/s\n\n";

    std::cout << "BLAS speedup: "
              << naive / blas
              << "x\n";

    std::cout << "C[0] = " << C[0] << "\n";
}