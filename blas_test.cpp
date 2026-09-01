#include <iostream>
#include <cblas.h>

int main()
{
    float A[] = {
        1, 2, 3,
        4, 5, 6};

    float B[] = {
        7, 8, 9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18};

    float C[2 * 4] = {};

    int M = 2;
    int K = 2;
    int N = 3;

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        M, N, K,
        1.0f,
        A, K,
        B, N,
        0.0f,
        C, N);

    for (int i = 0; i < M * N; i++)
        std::cout << C[i] << " ";
    std::cout << "\n";
}