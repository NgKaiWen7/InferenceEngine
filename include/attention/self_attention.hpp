#pragma once

#include "safetensors.hpp"
#include <stdfloat>
#include <string>
#include <torch/torch.h>
#include <vector>

struct TransformerWorkspace {
    torch::Tensor query;
    torch::Tensor key;
    torch::Tensor value;
    torch::Tensor context;
    torch::Tensor attention_dense;
    torch::Tensor intermediate;

    TransformerWorkspace(size_t sequence_length) {
        int64_t seq = static_cast<int64_t>(sequence_length);

        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

        query = torch::empty({seq, 1024}, options).to(torch::kCUDA);
        key = torch::empty({seq, 1024}, options).to(torch::kCUDA);
        value = torch::empty({seq, 1024}, options).to(torch::kCUDA);
        context = torch::empty({seq, 1024}, options).to(torch::kCUDA);

        attention_dense = torch::empty({seq, 1024}, options).to(torch::kCUDA);
        intermediate = torch::empty({seq, 4096}, options).to(torch::kCUDA);
    }
};

class TransformerLayer {
  public:
    void load(const std::string &file_path, int layer);
    torch::Tensor forward(const torch::Tensor &input, TransformerWorkspace &workspace);

  private:
    SafeTensorLoader tensor_loader;

    const int hidden_size = 1024;
    const int num_heads = 16;
    const int head_dim = 64;
    const float scaling = 1 / std::sqrt(static_cast<float>(head_dim));

    torch::Tensor attention_query_weight;
    torch::Tensor attention_query_bias;

    torch::Tensor attention_key_weight;
    torch::Tensor attention_key_bias;

    torch::Tensor attention_value_weight;
    torch::Tensor attention_value_bias;

    torch::Tensor attention_output_weight;
    torch::Tensor attention_output_bias;

    torch::Tensor attention_layernorm_weight;
    torch::Tensor attention_layernorm_bias;

    torch::Tensor intermediate_weight;
    torch::Tensor intermediate_bias;

    torch::Tensor output_weight;
    torch::Tensor output_bias;

    torch::Tensor output_layernorm_weight;
    torch::Tensor output_layernorm_bias;

    torch::nn::functional::LayerNormFuncOptions attention_norm_options{{1024}};
    torch::nn::functional::LayerNormFuncOptions output_norm_options{{1024}};
};
