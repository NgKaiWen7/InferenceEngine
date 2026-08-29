#pragma once

#include <string>
#include <vector>
#include "safetensors.hpp"
#include <stdfloat>

class TransformerLayer
{
public:
    void load(const std::string &file_path, int layer);
    void linear(
        const std::vector<std::vector<float>> &input,
        const uint16_t *weight,
        const uint16_t *bias,
        int input_size,
        int output_size,
        std::vector<std::vector<float>> &output);
    void layer_norm(
        const std::vector<std::vector<float>> &attention_dense,
        const uint16_t *weight,
        const uint16_t *bias,
        std::vector<std::vector<float>> &output);
    void attention(
        const std::vector<std::vector<float>> &input,
        std::vector<std::vector<float>> &output);

private:
    SafeTensorLoader tensor_loader;
    uint16_t *attention_query_weight;
    uint16_t *attention_query_bias;

    uint16_t *attention_key_weight;
    uint16_t *attention_key_bias;

    uint16_t *attention_value_weight;
    uint16_t *attention_value_bias;

    uint16_t *attention_output_weight;
    uint16_t *attention_output_bias;

    uint16_t *attention_layernorm_weight;
    uint16_t *attention_layernorm_bias;

    uint16_t *intermediate_weight;
    uint16_t *intermediate_bias;

    uint16_t *output_weight;
    uint16_t *output_bias;

    uint16_t *output_layernorm_weight;
    uint16_t *output_layernorm_bias;
};