#include <cblas.h>
#include <iostream>

int main()
{
    // Input: M x K = 2 x 2
    //
    // [1 2]
    // [3 4]
    float input[] = {
        1, 2,
        3, 4
    };

    // Weight: N x K = 3 x 2
    //
    // [1  2]
    // [3  4]
    // [5  6]
    float weight[] = {
        1, 2,
        3, 4,
        5, 6
    };

    // Bias: N = 3
    float bias[] = {
        10, 20, 30
    };

    // Output: M x N = 2 x 3
    //
    // [ ? ? ? ]
    // [ ? ? ? ]
    float output[2 * 3];

    size_t M = 2;
    size_t K = 2;
    size_t N = 3;

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M,
        N,
        K,
        1.0f,
        input,
        K,
        weight,
        K,
        0.0f,
        output,
        N
    );

    // Add bias
    // for (size_t i = 0; i < M; ++i)
    //     for (size_t j = 0; j < N; ++j)
    //         output[i * N + j] += bias[j];

    std::cout << "Input:\n";
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < K; ++j)
            std::cout << input[i * K + j] << " ";
        std::cout << '\n';
    }

    std::cout << "\nWeight:\n";
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < K; ++j)
            std::cout << weight[i * K + j] << " ";
        std::cout << '\n';
    }

    std::cout << "\nOutput:\n";
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
            std::cout << output[i * N + j] << " ";
        std::cout << '\n';
    }

    return 0;
}