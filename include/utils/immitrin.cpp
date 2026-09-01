#include <immintrin.h>
#include "utils/immitrin.hpp"
#include "attention/self_attention.hpp"
#include <cblas.h>

inline __m256 gelu_avx2(__m256 x)
{
    const __m256 c0 = _mm256_set1_ps(0.044715f);
    const __m256 c1 = _mm256_set1_ps(0.7978845608f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c27 = _mm256_set1_ps(27.0f);
    const __m256 c9 = _mm256_set1_ps(9.0f);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);

    __m256 inner = _mm256_add_ps(
        x,
        _mm256_mul_ps(c0, x3));

    inner = _mm256_mul_ps(c1, inner);

    __m256 inner2 = _mm256_mul_ps(inner, inner);

    __m256 numerator = _mm256_mul_ps(
        inner,
        _mm256_add_ps(c27, inner2));

    __m256 denominator = _mm256_add_ps(
        c27,
        _mm256_mul_ps(c9, inner2));

    __m256 tanh_value = _mm256_div_ps(numerator, denominator);

    return _mm256_mul_ps(
        _mm256_mul_ps(half, x),
        _mm256_add_ps(one, tanh_value));
}

inline float max_avx2(const float *x, size_t n)
{
    __m256 max_value = _mm256_set1_ps(-INFINITY);

    size_t i = 0;

    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(x + i);
        max_value = _mm256_max_ps(max_value, v);
    }

    alignas(32) float result[8];
    _mm256_store_ps(result, max_value);

    float max_result = result[0];

    for (int j = 1; j < 8; j++)
        max_result = std::max(max_result, result[j]);

    for (; i < n; i++)
        max_result = std::max(max_result, x[i]);

    return max_result;
}

float gelu_scalar(float x)
{
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}
void gelu(Tensor &values)
{
    size_t i = 0;

    for (; i + 8 <= values.size; i += 8)
    {
        __m256 x = _mm256_loadu_ps(values.data + i);
        __m256 y = gelu_avx2(x);
        _mm256_storeu_ps(values.data + i, y);
    }
    for (; i < values.size; ++i)
        values.data[i] = gelu_scalar(values.data[i]);
}

inline float sum_avx2(const float *x, size_t n)
{
    __m256 acc = _mm256_setzero_ps();

    size_t i = 0;

    for (; i + 8 <= n; i += 8)
    {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + i));
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, acc);

    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n; ++i)
    {
        sum += x[i];
    }
    return sum;
}

inline float sum_square_avx2(const float *x, size_t n)
{
    size_t i = 0;
    __m256 acc = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }

    float result[8];
    _mm256_storeu_ps(result, acc);

    float sum = result[0] + result[1] + result[2] + result[3] +
                result[4] + result[5] + result[6] + result[7];

    for (; i < n; ++i)
        sum += x[i] * x[i];

    return sum;
}

inline void sub_const(float *x, const float c, size_t n)
{
    size_t i = 0;
    const __m256 c_256 = _mm256_set1_ps(c);

    for (; i + 8 <= n; i += 8)
    {
        __m256 x_256 = _mm256_loadu_ps(x + i);
        __m256 output = _mm256_sub_ps(x_256, c_256);
        _mm256_storeu_ps(x + i, output);
    }

    for (; i < n; ++i)
        x[i] -= c;
}

inline void division(float *x, const float c, size_t n)
{
    size_t i = 0;
    const __m256 inv_c = _mm256_set1_ps(1.0f / c);

    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, inv_c));
    }

    for (; i < n; ++i)
        x[i] *= inv_c[0];
}

void residual(Tensor &current_layers, const Tensor &previous_layers)
{
    if (current_layers.size != previous_layers.size)
        throw std::invalid_argument("Residual tensors must have the same size");

    size_t i = 0;
    for (; i + 8 <= current_layers.size; i += 8)
    {
        __m256 current = _mm256_loadu_ps(current_layers.data + i);
        __m256 previous = _mm256_loadu_ps(previous_layers.data + i);
        __m256 new_data = _mm256_add_ps(current, previous);
        _mm256_storeu_ps(current_layers.data + i, new_data);
    }
    // Remaining elements
    for (; i < current_layers.size; ++i)
        current_layers.data[i] += previous_layers.data[i];
}

void layer_norm(Tensor &input, const Tensor &weight, const Tensor &bias)
{
    constexpr float eps = 1e-16f;

    size_t rows = input.shape[0];
    size_t cols = input.shape[1];

    for (size_t i = 0; i < rows; ++i)
    {
        float *row = input.data + i * cols;

        float sum = sum_avx2(row, cols);
        float mean = sum / cols;
        sub_const(row, mean, cols);

        float variance = sum_square_avx2(row, cols);
        variance /= cols;
        float inv_std = std::sqrt(variance + eps);

        division(row, inv_std, cols);
        size_t j = 0;
        for (; j + 8 <= cols; j += 8)
        {
            __m256 w = _mm256_loadu_ps(weight.data + j);
            __m256 b = _mm256_loadu_ps(bias.data + j);
            __m256 x = _mm256_loadu_ps(row + j);
            __m256 mul = _mm256_mul_ps(w, x);
            __m256 y = _mm256_add_ps(mul, b);
            _mm256_storeu_ps(row + j, y);
        }
        // Remaining elements
        for (; j < cols; ++j)
            row[j] = weight.data[j] * row[j] + bias.data[j];
    }
}

void normalize(float *x, size_t size)
{
    float norm = cblas_sdot(size, x, 1, x, 1);
    norm = std::sqrt(norm);
    division(x, norm, size);
}

void QKV(const Tensor &query, const Tensor &value, const Tensor &key,
         const int num_heads, const int head_dim, const int sequence_length, const int hidden_size, const float scaling,
         Tensor &scores, Tensor &context)
{
#pragma omp parallel for
    for (int h = 0; h < num_heads; h++)
    {
        int offset = h * head_dim;
        int score_offset = h * sequence_length * sequence_length;

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            sequence_length,
            sequence_length,
            head_dim,
            scaling,
            query.data + offset, hidden_size,
            key.data + offset, hidden_size,
            0.0f,
            scores.data + score_offset, sequence_length);

        for (size_t i = 0; i < sequence_length; i++)
        {
            float *row = scores.data + score_offset + i * sequence_length;
            float max_value = max_avx2(row, sequence_length);
            sub_const(row, max_value, sequence_length);
            for (size_t j = 0; j < sequence_length; ++j)
            {
                row[j] = std::exp(row[j]);
            }
            float sum = sum_avx2(row, sequence_length);
            division(row, sum, sequence_length);
        }

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            sequence_length,
            head_dim,
            sequence_length,
            1.0f,
            scores.data + score_offset, sequence_length,
            value.data + offset, hidden_size,
            0.0f,
            context.data + offset, hidden_size);
    }
}