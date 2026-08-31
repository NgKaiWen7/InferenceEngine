#include "attention/self_attention.hpp"
#include "safetensors.hpp"
#include "utils/conversion.hpp"
#include "utils/immitrin.hpp"
#include <stdfloat>
#include <cstdint>
#include <bit>
#include <cblas.h>

void TransformerLayer::load(const std::string &file_path, int layer)
{
    tensor_loader.load(file_path);

    std::string prefix = "encoder.layer." + std::to_string(layer) + ".";

    attention_query_weight = tensor_loader.get_tensor(prefix + "attention.self.query.weight");
    attention_query_bias = tensor_loader.get_tensor(prefix + "attention.self.query.bias");

    attention_key_weight = tensor_loader.get_tensor(prefix + "attention.self.key.weight");
    attention_key_bias = tensor_loader.get_tensor(prefix + "attention.self.key.bias");

    attention_value_weight = tensor_loader.get_tensor(prefix + "attention.self.value.weight");
    attention_value_bias = tensor_loader.get_tensor(prefix + "attention.self.value.bias");

    attention_output_weight = tensor_loader.get_tensor(prefix + "attention.output.dense.weight");
    attention_output_bias = tensor_loader.get_tensor(prefix + "attention.output.dense.bias");

    attention_layernorm_weight = tensor_loader.get_tensor(prefix + "attention.output.LayerNorm.weight");
    attention_layernorm_bias = tensor_loader.get_tensor(prefix + "attention.output.LayerNorm.bias");

    intermediate_weight = tensor_loader.get_tensor(prefix + "intermediate.dense.weight");
    intermediate_bias = tensor_loader.get_tensor(prefix + "intermediate.dense.bias");

    output_weight = tensor_loader.get_tensor(prefix + "output.dense.weight");
    output_bias = tensor_loader.get_tensor(prefix + "output.dense.bias");

    output_layernorm_weight = tensor_loader.get_tensor(prefix + "output.LayerNorm.weight");
    output_layernorm_bias = tensor_loader.get_tensor(prefix + "output.LayerNorm.bias");
}
void TransformerLayer::linear(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    Tensor &output)
{
    size_t M = input.shape[0];
    size_t K = input.shape[1];
    size_t N = weight.shape[0];

    // std::cout
    //     << "Linear: "
    //     << "input=[" << M << ", " << K << "] "
    //     << "weight=[" << weight.shape[0] << ", " << weight.shape[1] << "] "
    //     << "bias=[" << bias.shape[0] << "] "
    //     << "output=[" << output.shape[0] << ", " << output.shape[1] << "] "
    //     << "GEMM=(" << M << "x" << K << ") * ("
    //     << K << "x" << N << ")"
    //     << std::endl;

    // auto gemm_start = std::chrono::high_resolution_clock::now();

    // gemm_avx2(input.data, weight.data, output.data, M, K, N);
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M, N, K,
        1.0f,
        input.data, K,
        weight.data, K,
        0.0f,
        output.data, N);

    auto gemm_end = std::chrono::high_resolution_clock::now();

    // double gemm_ms =
    //     std::chrono::duration<double, std::milli>(
    //         gemm_end - gemm_start)
    //         .count();

    // auto bias_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            output.data[i * N + j] += bias.data[j];

    // auto bias_end = std::chrono::high_resolution_clock::now();

    // double bias_ms =
    //     std::chrono::duration<double, std::milli>(
    //         bias_end - bias_start)
    //         .count();

    // std::cout
    //     << "  GEMM: " << gemm_ms << " ms"
    //     << " | Bias: " << bias_ms << " ms"
    //     << std::endl;
}

inline float dot_product_avx2(const float *a, const float *b, size_t n)
{
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;

    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    float total = sum[0] + sum[1] + sum[2] + sum[3] +
                  sum[4] + sum[5] + sum[6] + sum[7];

    for (; i < n; i++)
        total += a[i] * b[i];

    return total;
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
void TransformerLayer::attention(const Tensor &input, Tensor &output, TransformerWorkspace &workspace)
{
    size_t sequence_length = input.shape[0];
    Tensor &query = workspace.query;
    Tensor &value = workspace.value;
    Tensor &key = workspace.key;
    Tensor &value_T = workspace.value_T;
    linear(input, attention_value_weight, attention_value_bias, value);
    linear(input, attention_query_weight, attention_query_bias, query);
    linear(input, attention_key_weight, attention_key_bias, key);
    
    Tensor &scores = workspace.scores;
    Tensor &context = workspace.context;
    #pragma omp parallel for collapse(2)
        for (int h = 0; h < num_heads; h++)
        {
            size_t offset = h * head_dim;
            size_t vt_offset = h * head_dim * sequence_length;
    
            for (size_t j = 0; j < sequence_length; j++)
            {
                const float *v = value.data + j * hidden_size + offset;
    
                for (int k = 0; k < head_dim; k++)
                {
                    value_T.data[vt_offset + k * sequence_length + j] = v[k];
                }
            }
        }

    #pragma omp parallel for collapse(2)
    for (int h = 0; h < num_heads; h++)
    {
        int offset = h * head_dim;
        int score_offset = h * sequence_length * sequence_length;

        for (size_t i = 0; i < sequence_length; i++)
        {
            for (size_t j = 0; j < sequence_length; j++)
            {
                float sum = dot_product_avx2(query.data + offset + i * hidden_size, key.data + offset + j * hidden_size, head_dim);
                scores.data[score_offset + i * sequence_length + j] = sum * scaling;
            }
        }
    }

#pragma omp parallel for
    for (int h = 0; h < num_heads; h++)
    {
        int offset = h * head_dim;
        int score_offset = h * sequence_length * sequence_length;
        for (size_t i = 0; i < sequence_length; i++)
        {
            float *row = scores.data + score_offset + i * sequence_length;
            float max_value = max_avx2(row, sequence_length);
            sub_const(row, max_value, sequence_length);
            float sum = 0.0f;

            for (size_t j = 0; j < sequence_length; ++j)
            {
                row[j] = std::exp(row[j]);
            }
            sum = sum_avx2(row, sequence_length);
            division(row, sum, sequence_length);
        }
    }

#pragma omp parallel for
    for (int h = 0; h < num_heads; h++)
    {
        int offset = h * head_dim;
        int score_offset = h * sequence_length * sequence_length;
        // Attention × V
        for (size_t i = 0; i < sequence_length; i++)
        {
            for (int k = 0; k < head_dim; k++)
            {
                float sum = dot_product_avx2(scores.data + score_offset + i * sequence_length, value_T.data + (offset + k) * sequence_length, sequence_length);
                context.data[hidden_size * i + offset + k] = sum;
            }
        }
    }

    Tensor &attention_dense = workspace.attention_dense;
    linear(context, attention_output_weight, attention_output_bias, attention_dense);
    
    residual(attention_dense, input);
    
    layer_norm(attention_dense, attention_layernorm_weight, attention_layernorm_bias);
    
    Tensor &intermediate = workspace.intermediate;
    linear(attention_dense, intermediate_weight, intermediate_bias, intermediate);

    #pragma omp parallel for
    for (size_t i = 0; i < intermediate.size; i++){
        intermediate.data[i] = 0.5f * intermediate.data[i] * (1.0f + std::erf(intermediate.data[i] * 0.7071067811865475f));
    }
    // gelu(intermediate);
    
    linear(intermediate, output_weight, output_bias, output);
    
    residual(output, attention_dense);
    
    layer_norm(output, output_layernorm_weight, output_layernorm_bias);
}