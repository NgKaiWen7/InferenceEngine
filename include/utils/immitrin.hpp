#include <immintrin.h>
#include "attention/self_attention.hpp"

float sum_avx2(const float *x, size_t n);
void gelu(Tensor &values);
void residual(Tensor &current_layers, const Tensor &previous_layers);
void layer_norm(Tensor &input, const Tensor &weight, const Tensor &bias);
void gemm_avx2(const float *A, const float *B, float *C, size_t M, size_t K, size_t N);
void normalize(float *x, size_t size);
void QKV(const Tensor &query, const Tensor &value, const Tensor &key,
         const int num_heads, const int head_dim, const int sequence_length, const int hidden_size, const float scaling,
         Tensor &scores, Tensor &context);