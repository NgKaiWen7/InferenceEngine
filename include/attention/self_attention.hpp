#pragma once

#include <string>
#include <vector>
#include "safetensors.hpp"
#include <stdfloat>

class TransformerLayer
{
public:
    void load(const std::string &file_path, int layer);
    void linear(const Tensor &input, const Tensor &weight, const Tensor &bias, Tensor &output);
    void layer_norm(const Tensor &attention_dense, const Tensor &weight, const Tensor &bias, Tensor &output);
    void attention(const Tensor &input, Tensor &output);

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