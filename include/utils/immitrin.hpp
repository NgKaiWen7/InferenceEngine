#include <immintrin.h>
#include "attention/self_attention.hpp"

float sum_avx2(const float *x, size_t n);
void gelu(Tensor &values);
void residual(Tensor &current_layers, const Tensor &previous_layers);
void layer_norm(Tensor &input, const Tensor &weight, const Tensor &bias);
void gemm_avx2(const float *A, const float *B, float *C, size_t M, size_t K, size_t N);
void normalize(float *x, size_t size);