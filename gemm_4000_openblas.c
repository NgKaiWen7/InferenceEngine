/* gemm_4000_openblas.c
 *
 * Single 4000x4000x4000 SGEMM using OpenBLAS via the generic CBLAS header.
 *
 * Build:
 *   gcc -O2 -march=native gemm_4000_openblas.c -o gemm_4000_openblas \
 *       -lopenblas -lpthread -lm
 *   OPENBLAS_NUM_THREADS=1 ./gemm_4000_openblas      # single-thread, fair vs mkl_sequential
 *   ./gemm_4000_openblas                             # default threading, uses all cores
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>   /* generic Netlib CBLAS surface, backed by OpenBLAS */

#define N 4000

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    const int m = N, n = N, k = N, lda = N, ldb = N, ldc = N;
    const float alpha = 1.0f, beta = 0.0f;

    printf("Allocating %.1f MB per matrix...\n", (double)N * N * sizeof(float) / 1e6);
    float *A = (float *)malloc(sizeof(float) * (size_t)m * k);
    float *B = (float *)malloc(sizeof(float) * (size_t)k * n);
    float *C = (float *)malloc(sizeof(float) * (size_t)m * n);
    if (!A || !B || !C) { fprintf(stderr, "allocation failed\n"); return 1; }

    srand(0);
    for (size_t i = 0; i < (size_t)m * k; ++i) A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)k * n; ++i) B[i] = (float)rand() / RAND_MAX;

    printf("Running OpenBLAS cblas_sgemm on %dx%dx%d...\n", N, N, N);
    double t0 = now_sec();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    double dt = now_sec() - t0;

    double gflops = 2.0 * (double)N * N * N / dt / 1e9;
    printf("OpenBLAS cblas_sgemm : %.3f s   (%.2f GFLOP/s)\n", dt, gflops);

    free(A); free(B); free(C);
    return 0;
}
