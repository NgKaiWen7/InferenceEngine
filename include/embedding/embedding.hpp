#pragma once

#include <string>
#include <vector>
#include "safetensors.hpp"
#include <stdfloat>

class Embedding
{
public:
    void load(const std::string file_path);
    void encode(const std::vector<int> &token_ids, Tensor &embeddings);

private:
    int embedding_dim;
    int vocab_size;

    int position_embedding_dim;
    int position_context_size;

    int token_type_dim;
    int token_type_size;

    SafeTensorLoader tensor_loader;
    Tensor embedding_weights;
    Tensor position_weights;
    Tensor token_type_weights;

    Tensor layernorm_weight;
    Tensor layernorm_bias;
};