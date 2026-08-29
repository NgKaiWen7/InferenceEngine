#include "pooler/pooler.hpp"
#include "safetensors.hpp"
#include <stdfloat>
#include <cstdint>
#include <bit>

void Pooler::load(const std::string file_path)
{
    tensor_loader.load(file_path);

    Tensor bias_tensor = tensor_loader.get_tensor("pooler.dense.bias");
    pooler_bias = reinterpret_cast<uint16_t *>(bias_tensor.data);

    Tensor weight_tensor = tensor_loader.get_tensor("pooler.dense.weight");
    pooler_weights = reinterpret_cast<uint16_t *>(weight_tensor.data);
}

void Pooler::linear(
    const std::vector<float> &input,
    const uint16_t *weight,
    const uint16_t *bias,
    int input_size,
    int output_size,
    std::vector<float> &output)
{
    output.resize(output_size);

    for (int j = 0; j < output_size; j++)
    {
        float sum = 0.0f;

        for (int k = 0; k < input_size; k++)
        {
            float w = static_cast<float>(
                std::bit_cast<std::float16_t>(
                    weight[j * input_size + k]));

            sum += input[k] * w;
        }

        float b = static_cast<float>(
            std::bit_cast<std::float16_t>(bias[j]));

        output[j] = sum + b;
    }
}
void Pooler::pool(
    const std::vector<std::vector<float>> &encoder_output,
    std::vector<std::vector<float>> &output)
{
    output.clear();
    output.reserve(encoder_output.size());

    for (const auto &row : encoder_output)
    {
        std::vector<float> pooled;

        linear(
            row,
            pooler_weights,
            pooler_bias,
            1024,
            1024,
            pooled);

        for (float &x : pooled)
            x = std::tanh(x);

        output.push_back(std::move(pooled));
    }
}