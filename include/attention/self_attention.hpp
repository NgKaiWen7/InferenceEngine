#pragma once

#include <string>
#include <vector>
#include "safetensors.hpp"
#include <stdfloat>

struct TransformerWorkspace
{
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor scores;
    Tensor context;
    Tensor attention_dense;
    Tensor intermediate;

    Tensor buffer_a;
    Tensor buffer_b;

    TransformerWorkspace(size_t sequence_length)
    {
        query.allocate({static_cast<int64_t>(sequence_length), 1024});
        key.allocate({static_cast<int64_t>(sequence_length), 1024});
        value.allocate({static_cast<int64_t>(sequence_length), 1024});
        context.allocate({static_cast<int64_t>(sequence_length), 1024});

        scores.allocate({16, static_cast<int64_t>(sequence_length), static_cast<int64_t>(sequence_length)});

        attention_dense.allocate({static_cast<int64_t>(sequence_length), 1024});
        intermediate.allocate({static_cast<int64_t>(sequence_length), 4096});

        buffer_a.allocate({static_cast<int64_t>(sequence_length), 1024});
        buffer_b.allocate({static_cast<int64_t>(sequence_length), 1024});
    }
};

class TransformerLayer
{
public:
    void load(const std::string &file_path, int layer);
    void linear(const Tensor &input, const Tensor &weight, const Tensor &bias, Tensor &output);
    void attention(const Tensor &input, Tensor &output, TransformerWorkspace &workspace);

private:
    SafeTensorLoader tensor_loader;

    const int hidden_size = 1024;
    const int num_heads = 16;
    const int head_dim = 64;

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
