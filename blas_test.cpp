#include <cblas.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cstdint>
#include <vector>

struct Tensor
{
    float *data = nullptr;
    size_t size = 0;
    std::vector<int64_t> shape;
};

void linear(const Tensor &input, const Tensor &weight, const Tensor &bias, Tensor &output)
{
    size_t M = input.shape[0];
    size_t K = input.shape[1];
    size_t N = weight.shape[0];

    auto start = std::chrono::steady_clock::now();

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M, N, K,
        1.0f,
        input.data, K,
        weight.data, K,
        0.0f,
        output.data, N
    );

    auto end = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    std::cout << "GEMM: "
              << ms << " ms | "
              << gflops << " GFLOP/s\n";
}

Tensor make_tensor(size_t rows, size_t cols)
{
    Tensor t;

    t.shape = {
        static_cast<int64_t>(rows),
        static_cast<int64_t>(cols)
    };

    t.size = rows * cols;
    t.data = new float[t.size];

    for (size_t i = 0; i < t.size; ++i)
        t.data[i] = 0.01f;

    return t;
}

Tensor make_vector(size_t size)
{
    Tensor t;

    t.shape = {static_cast<int64_t>(size)};
    t.size = size;
    t.data = new float[t.size];

    for (size_t i = 0; i < t.size; ++i)
        t.data[i] = 0.01f;

    return t;
}

int main()
{
    const size_t M = 130;
    const size_t K = 1024;
    const size_t N = 1024;

    Tensor input = make_tensor(M, K);
    Tensor weight = make_tensor(N, K);
    Tensor bias = make_vector(N);
    Tensor output = make_tensor(M, N);

    std::cout
        << "Input:  [" << input.shape[0] << ", " << input.shape[1] << "]\n"
        << "Weight: [" << weight.shape[0] << ", " << weight.shape[1] << "]\n"
        << "Bias:   [" << bias.shape[0] << "]\n"
        << "Output: [" << output.shape[0] << ", " << output.shape[1] << "]\n";

    std::cout << "\nMemory:\n";

    std::cout
        << "input  = " << static_cast<void*>(input.data)
        << " alignment=" << (reinterpret_cast<uintptr_t>(input.data) % 64)
        << '\n';

    std::cout
        << "weight = " << static_cast<void*>(weight.data)
        << " alignment=" << (reinterpret_cast<uintptr_t>(weight.data) % 64)
        << '\n';

    std::cout
        << "bias   = " << static_cast<void*>(bias.data)
        << " alignment=" << (reinterpret_cast<uintptr_t>(bias.data) % 64)
        << '\n';

    std::cout
        << "output = " << static_cast<void*>(output.data)
        << " alignment=" << (reinterpret_cast<uintptr_t>(output.data) % 64)
        << '\n';

    std::cout << "\nWarmup:\n";

    for (int i = 0; i < 10; ++i)
        linear(input, weight, bias, output);

    std::cout << "\nBenchmark:\n";

    for (int i = 0; i < 20; ++i)
    {
        std::cout << i << ": ";
        linear(input, weight, bias, output);
    }

    delete[] input.data;
    delete[] weight.data;
    delete[] bias.data;
    delete[] output.data;

    return 0;
}