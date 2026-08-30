#include "attention/self_attention.hpp"
#include "safetensors.hpp"
#include "utils/conversion.hpp"
#include <stdfloat>
#include <cstdint>
#include <bit>
#include <cblas.h>

float gelu(float x)
{
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

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

// void TransformerLayer::linear(
//     const Tensor input,
//     const Tensor weight,
//     const Tensor bias,
//     Tensor output)
// {
//     for (size_t i = 0; i < input.size(); i++)
//     {
//         std::vector<float> row(output_size);

//         for (int j = 0; j < output_size; j++)
//         {
//             float sum = 0.0f;

//             for (int k = 0; k < input_size; k++)
//             {
//                 float w = weight.data[j * input_size + k];

//                 sum += input[i][k] * w;
//             }

//             float b = bias.data[j];

//             row[j] = sum + b;
//         }

//         output.push_back(std::move(row));
//     }
// }

void TransformerLayer::linear(const Tensor &input, const Tensor &weight, const Tensor &bias, Tensor &output)
{
    size_t M = input.shape[0];
    size_t K = input.shape[1];
    size_t N = weight.shape[0];
    output.size = M * N;
    output.shape = {static_cast<int64_t>(M), static_cast<int64_t>(N)};
    output.data = new float[output.size];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, input.data, K, weight.data, K, 0.0f, output.data, N);

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            output.data[i * N + j] += bias.data[j];
}

void TransformerLayer::layer_norm(const Tensor &input, const Tensor &weight, const Tensor &bias, Tensor &output)
{
    constexpr float eps = 1e-5f;

    size_t rows = input.shape[0];
    size_t cols = input.shape[1];

    output.size = input.size;
    output.shape = input.shape;
    output.data = new float[output.size];

    for (size_t i = 0; i < rows; ++i)
    {
        const float *row = input.data + i * cols;

        float mean = 0.0f;

        for (size_t j = 0; j < cols; ++j)
            mean += row[j];

        mean /= cols;

        float variance = 0.0f;

        for (size_t j = 0; j < cols; ++j)
        {
            float diff = row[j] - mean;
            variance += diff * diff;
        }

        variance /= cols;

        float inv_std = 1.0f / std::sqrt(variance + eps);

        for (size_t j = 0; j < cols; ++j)
            output.data[i * cols + j] = (row[j] - mean) * inv_std * weight.data[j] + bias.data[j];
    }
}

void TransformerLayer::attention(const Tensor &input, Tensor &output)
{
    constexpr int hidden_size = 1024;
    constexpr int num_heads = 16;
    constexpr int head_dim = 64;

    size_t sequence_length = input.shape[0];

    Tensor value;
    linear(input, attention_value_weight, attention_value_bias, value);
    Tensor query;
    linear(input, attention_query_weight, attention_query_bias, query);
    Tensor key;
    linear(input, attention_key_weight, attention_key_bias, key);

    Tensor context;
    context.shape = {static_cast<int64_t>(sequence_length), hidden_size};
    context.size = sequence_length * hidden_size;
    context.data = new float[context.size];

    #pragma omp parallel for
    for (int h = 0; h < num_heads; h++)
    {
        int offset = h * head_dim;

        Tensor scores;
        scores.shape = {static_cast<int64_t>(sequence_length), static_cast<int64_t>(sequence_length)};
        scores.size = sequence_length * sequence_length;
        scores.data = new float[scores.size];

        for (size_t i = 0; i < sequence_length; i++)
        {
            for (size_t j = 0; j < sequence_length; j++)
            {
                float sum = 0.0f;

                for (int k = 0; k < head_dim; k++)
                {
                    sum += query.data[offset + i * hidden_size + k] * key.data[offset + j * hidden_size + k];
                }
                scores.data[i * sequence_length + j] = sum / std::sqrt(static_cast<float>(head_dim));
            }
        }

        // Softmax
        for (size_t i = 0; i < sequence_length; ++i)
        {
            float *row = scores.data + i * sequence_length;

            float max_value = row[0];

            for (size_t j = 1; j < sequence_length; ++j)
                max_value = std::max(max_value, row[j]);

            float sum = 0.0f;

            for (size_t j = 0; j < sequence_length; ++j)
            {
                row[j] = std::exp(row[j] - max_value);
                sum += row[j];
            }

            for (size_t j = 0; j < sequence_length; ++j)
                row[j] /= sum;
        }

        // Attention × V
        for (size_t i = 0; i < sequence_length; i++)
        {
            for (int k = 0; k < head_dim; k++)
            {
                float sum = 0.0f;

                for (size_t j = 0; j < sequence_length; j++)
                    sum += scores.data[i * sequence_length + j] * value.data[offset + j * hidden_size + k];
                context.data[offset + i * hidden_size + k] = sum;
            }
        }
    }

    Tensor attention_dense;
    linear(context, attention_output_weight, attention_output_bias, attention_dense);

    Tensor residual;
    residual.size = attention_dense.size;
    residual.shape = attention_dense.shape;
    residual.data = new float[residual.size];

    for (size_t i = 0; i < residual.size; ++i)
        residual.data[i] = attention_dense.data[i] + input.data[i];

    Tensor attention_output;
    layer_norm(residual, attention_layernorm_weight, attention_layernorm_bias, attention_output);

    Tensor intermediate;
    linear(attention_output, intermediate_weight, intermediate_bias, intermediate);

    for (size_t i = 0; i < intermediate.size; ++i)
        intermediate.data[i] = gelu(intermediate.data[i]);

    Tensor output_dense;
    linear(intermediate, output_weight, output_bias, output_dense);

    Tensor output_residual;
    output_residual.size = output_dense.size;
    output_residual.shape = output_dense.shape;
    output_residual.data = new float[output_residual.size];

    for (size_t i = 0; i < output_residual.size; ++i)
        output_residual.data[i] = output_dense.data[i] + attention_output.data[i];

    layer_norm(output_residual, output_layernorm_weight, output_layernorm_bias, output);

}