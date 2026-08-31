#include <cblas.h>
#include <chrono>
#include <iostream>
#include <vector>

double bench(size_t M, size_t K, size_t N, int iterations)
{
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(N * K, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M, N, K,
            1.0f,
            A.data(), K,
            B.data(), K,
            0.0f,
            C.data(), N);
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start).count() / iterations;
}

int main()
{
    constexpr int iterations = 100;

    std::cout << "FFN 1: "
              << bench(78, 1024, 4096, iterations)
              << " ms\n";

    std::cout << "FFN 2: "
              << bench(78, 4096, 1024, iterations)
              << " ms\n";
}