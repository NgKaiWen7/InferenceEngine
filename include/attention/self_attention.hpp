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
        const Tensor weight,
        const Tensor bias,
        int input_size,
        int output_size,
        std::vector<std::vector<float>> &output);
    void layer_norm(
        const std::vector<std::vector<float>> &attention_dense,
        const Tensor weight,
        const Tensor bias,
        std::vector<std::vector<float>> &output);
    void attention(
        const std::vector<std::vector<float>> &input,
        std::vector<std::vector<float>> &output);

private:
    SafeTensorLoader tensor_loader;
    Tensor attention_query_weight;
    Tensor attention_query_bias;

    Tensor attention_key_weight;
    Tensor attention_key_bias;

    Tensor attention_value_weight;
    Tensor attention_value_bias;

    Tensor attention_output_weight;
    Tensor attention_output_bias;

    Tensor attention_layernorm_weight;
    Tensor attention_layernorm_bias;

    Tensor intermediate_weight;
    Tensor intermediate_bias;

    Tensor output_weight;
    Tensor output_bias;

    Tensor output_layernorm_weight;
    Tensor output_layernorm_bias;
};