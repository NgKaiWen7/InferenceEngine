/* gemm_4000_mkl.c
 *
 * Single 4000x4000x4000 SGEMM using MKL's CBLAS interface.
 *
 * Build (pip-installed MKL, as set up earlier in this session):
 *   gcc -O2 -march=native gemm_4000_mkl.c -o gemm_4000_mkl \
 *       -l:libmkl_intel_lp64.so.3 -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 \
 *       -lpthread -lm -ldl
 *   LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./gemm_4000_mkl
 *
 * Build (apt oneAPI install):
 *   gcc -O2 -march=native gemm_4000_mkl.c -o gemm_4000_mkl \
 *       -I"${MKLROOT}/include" -L"${MKLROOT}/lib/intel64" \
 *       -Wl,-rpath,"${MKLROOT}/lib/intel64" \
 *       -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
 *   ./gemm_4000_mkl
 *
 * Note: this links mkl_sequential for a like-for-like comparison against
 * OpenBLAS run with OPENBLAS_NUM_THREADS=1. On a real multi-core machine,
 * swap in -lmkl_intel_thread -liomp5 (and drop -lmkl_sequential) to let MKL
 * thread the GEMM internally -- at 4000x4000 that will matter a lot.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>   /* _mm_malloc / _mm_free -- 64-byte aligned buffers */
#include <mkl.h>         /* cblas_sgemm */

#define N 4000

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    const MKL_INT m = N, n = N, k = N, lda = N, ldb = N, ldc = N;
    const float alpha = 1.0f, beta = 0.0f;

    printf("Allocating %.1f MB per matrix...\n", (double)N * N * sizeof(float) / 1e6);
    float *A = (float *)_mm_malloc(sizeof(float) * (size_t)m * k, 64);
    float *B = (float *)_mm_malloc(sizeof(float) * (size_t)k * n, 64);
    float *C = (float *)_mm_malloc(sizeof(float) * (size_t)m * n, 64);
    if (!A || !B || !C) { fprintf(stderr, "allocation failed\n"); return 1; }

    srand(0);
    for (size_t i = 0; i < (size_t)m * k; ++i) A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)k * n; ++i) B[i] = (float)rand() / RAND_MAX;

    printf("Running MKL cblas_sgemm on %dx%dx%d...\n", N, N, N);
    double t0 = now_sec();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    double dt = now_sec() - t0;

    double gflops = 2.0 * (double)N * N * N / dt / 1e9;
    printf("MKL cblas_sgemm : %.3f s   (%.2f GFLOP/s)\n", dt, gflops);

    _mm_free(A); _mm_free(B); _mm_free(C);
    return 0;
}
