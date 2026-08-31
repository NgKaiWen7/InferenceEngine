#include <immintrin.h>
#include "utils/immitrin.hpp"
#include "attention/self_attention.hpp"

inline __m256 gelu_avx2(__m256 x)
{
    const __m256 c0 = _mm256_set1_ps(0.044715f);
    const __m256 c1 = _mm256_set1_ps(0.7978845608f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);

    __m256 inner = _mm256_add_ps(x, _mm256_mul_ps(c0, x3));

    inner = _mm256_mul_ps(c1, inner);

    __m256 inner2 = _mm256_mul_ps(inner, inner);

    __m256 numerator = _mm256_mul_ps(inner, _mm256_add_ps(_mm256_set1_ps(27.0f), inner2));
    __m256 denominator = _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), inner2));
    __m256 tanh_value = _mm256_div_ps(numerator, denominator);

    return _mm256_mul_ps(
        _mm256_mul_ps(half, x),
        _mm256_add_ps(one, tanh_value));
}

void gelu(Tensor &values)
{
    for (size_t i = 0; i < values.size; i += 8)
    {
        __m256 x = _mm256_loadu_ps(values.data + i);
        __m256 y = gelu_avx2(x);
        _mm256_storeu_ps(values.data + i, y);
    }
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
    const __m256 c_256 = _mm256_set1_ps(c);

    for (; i + 8 <= n; i += 8)
    {
        __m256 x_256 = _mm256_loadu_ps(x + i);
        __m256 output = _mm256_div_ps(x_256, c_256);
        _mm256_storeu_ps(x + i, output);
    }

    for (; i < n; ++i)
        x[i] = x[i] / c;
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
    constexpr float eps = 1e-5f;

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
        float inv_std = 1.0f / std::sqrt(variance + eps);

        division(row, inv_std, cols);
        size_t j = 0;
        for (; j + 8 <= cols; j += 8)
        {
            __m256 w = _mm256_loadu_ps(weight.data + i);
            __m256 b = _mm256_loadu_ps(bias.data + i);
            __m256 x = _mm256_loadu_ps(row + i);
            __m256 mul = _mm256_mul_ps(w, x);
            __m256 y = _mm256_add_ps(mul, b);
            _mm256_storeu_ps(row + i, y);
        }
        // Remaining elements
        for (; i < cols; ++i)
            row[i] = weight.data[i] * row[i] + bias.data[i];
    }
}