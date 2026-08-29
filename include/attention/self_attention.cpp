#include "attention/self_attention.hpp"
#include "safetensors.hpp"
#include "safetensors.hpp"
#include <stdfloat>
#include <cstdint>
#include <bit>

float gelu(float x)
{
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

void TransformerLayer::load(const std::string &file_path, int layer)
{
    tensor_loader.load(file_path);

    std::string prefix = "encoder.layer." + std::to_string(layer) + ".";

    Tensor t;

    t = tensor_loader.get_tensor(prefix + "attention.self.query.weight");
    attention_query_weight = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.self.query.bias");
    attention_query_bias = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.self.key.weight");
    attention_key_weight = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.self.key.bias");
    attention_key_bias = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.self.value.weight");
    attention_value_weight = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.self.value.bias");
    attention_value_bias = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.output.dense.weight");
    attention_output_weight = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.output.dense.bias");
    attention_output_bias = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.output.LayerNorm.weight");
    attention_layernorm_weight = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "attention.output.LayerNorm.bias");
    attention_layernorm_bias = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "intermediate.dense.weight");
    intermediate_weight = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "intermediate.dense.bias");
    intermediate_bias = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "output.dense.weight");
    output_weight = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "output.dense.bias");
    output_bias = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "output.LayerNorm.weight");
    output_layernorm_weight = reinterpret_cast<uint16_t *>(t.data);

    t = tensor_loader.get_tensor(prefix + "output.LayerNorm.bias");
    output_layernorm_bias = reinterpret_cast<uint16_t *>(t.data);
}

void TransformerLayer::linear(
    const std::vector<std::vector<float>> &input,
    const uint16_t *weight,
    const uint16_t *bias,
    int input_size,
    int output_size,
    std::vector<std::vector<float>> &output)
{
    for (size_t i = 0; i < input.size(); i++)
    {
        std::vector<float> row(output_size);

        for (int j = 0; j < output_size; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < input_size; k++)
            {
                float w = static_cast<float>(
                    std::bit_cast<std::float16_t>(
                        weight[j * input_size + k]));

                sum += input[i][k] * w;
            }

            float b = static_cast<float>(
                std::bit_cast<std::float16_t>(bias[j]));

            row[j] = sum + b;
        }

        output.push_back(std::move(row));
    }
}

void TransformerLayer::layer_norm(
    const std::vector<std::vector<float>> &input,
    const uint16_t *weight,
    const uint16_t *bias,
    std::vector<std::vector<float>> &output)
{
    constexpr float eps = 1e-5f;

    for (const auto &row : input)
    {
        float mean = 0.0f;

        for (float x : row)
            mean += x;

        mean /= row.size();

        float variance = 0.0f;

        for (float x : row)
        {
            float diff = x - mean;
            variance += diff * diff;
        }

        variance /= row.size();

        float inv_std = 1.0f / std::sqrt(variance + eps);

        std::vector<float> normalized(row.size());

        for (size_t j = 0; j < row.size(); ++j)
        {
            float gamma = static_cast<float>(
                std::bit_cast<std::float16_t>(weight[j]));

            float beta = static_cast<float>(
                std::bit_cast<std::float16_t>(bias[j]));

            normalized[j] =
                (row[j] - mean) * inv_std * gamma + beta;
        }

        output.push_back(std::move(normalized));
    }
}

void TransformerLayer::attention(
    const std::vector<std::vector<float>> &input,
    std::vector<std::vector<float>> &output)
{
    constexpr int hidden_size = 1024;
    constexpr int num_heads = 16;
    constexpr int head_dim = 64;
    size_t sequence_length = input.size();

    std::vector<std::vector<float>> value;
    linear(input, attention_value_weight, attention_value_bias, 1024, 1024, value);
    std::vector<std::vector<float>> query;
    linear(input, attention_query_weight, attention_query_bias, 1024, 1024, query);
    std::vector<std::vector<float>> key;
    linear(input, attention_key_weight, attention_key_bias, 1024, 1024, key);

    std::vector<std::vector<float>> context(
        sequence_length,
        std::vector<float>(hidden_size));

    for (int h = 0; h < num_heads; h++)
    {
        int offset = h * head_dim;

        // Attention scores: [N, N]
        std::vector<std::vector<float>> scores(
            sequence_length,
            std::vector<float>(sequence_length));

        for (size_t i = 0; i < sequence_length; i++)
        {
            for (size_t j = 0; j < sequence_length; j++)
            {
                float sum = 0.0f;

                for (int k = 0; k < head_dim; k++)
                {
                    sum += query[i][offset + k] *
                           key[j][offset + k];
                }

                scores[i][j] = sum / std::sqrt(static_cast<float>(head_dim));
            }
        }

        // Softmax
        for (size_t i = 0; i < sequence_length; i++)
        {
            float sum = 0.0f;
            for (size_t j = 0; j < sequence_length; j++)
            {
                scores[i][j] = std::exp(scores[i][j]);
                sum += scores[i][j];
            }

            for (size_t j = 0; j < sequence_length; j++)
                scores[i][j] /= sum;
        }

        // Attention × V
        for (size_t i = 0; i < sequence_length; i++)
        {
            for (int k = 0; k < head_dim; k++)
            {
                float sum = 0.0f;

                for (size_t j = 0; j < sequence_length; j++)
                    sum += scores[i][j] * value[j][offset + k];

                context[i][offset + k] = sum;
            }
        }
    }
    std::vector<std::vector<float>> attention_dense;
    linear(
        context,
        attention_output_weight,
        attention_output_bias,
        1024, 1024,
        attention_dense);

    std::vector<std::vector<float>> residual = attention_dense;
    for (size_t i = 0; i < residual.size(); i++)
    {
        for (int j = 0; j < 1024; ++j)
            residual[i][j] += input[i][j];
    }

    std::vector<std::vector<float>> attention_output;
    layer_norm(
        residual,
        attention_layernorm_weight,
        attention_layernorm_bias,
        attention_output);

    std::vector<std::vector<float>> intermediate;
    linear(
        attention_output,
        intermediate_weight,
        intermediate_bias,
        1024, 4096,
        intermediate);
    for (auto &row : intermediate)
    {
        for (float &x : row)
            x = gelu(x);
    }

    std::vector<std::vector<float>> output_dense;
    linear(
        intermediate,
        output_weight,
        output_bias,
        4096, 1024,
        output_dense);

    std::vector<std::vector<float>> output_residual = output_dense;
    for (size_t i = 0; i < residual.size(); i++)
    {
        for (int j = 0; j < 1024; ++j)
            output_residual[i][j] += attention_output[i][j];
    }
    layer_norm(
        output_residual,
        output_layernorm_weight,
        output_layernorm_bias,
        output);

}