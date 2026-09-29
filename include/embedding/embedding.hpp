#pragma once

#include "safetensors.hpp"
#include <stdfloat>
#include <string>
#include <torch/torch.h>
#include <vector>

class Embedding {
  public:
    void load(const std::string file_path);
    void encode(const std::vector<int> &token_ids, torch::Tensor &embeddings);

  private:
    int embedding_dim;
    int vocab_size;

    int position_embedding_dim;
    int position_context_size;

    int token_type_dim;
    int token_type_size;

    SafeTensorLoader tensor_loader;
    torch::Tensor embedding_weights;
    torch::Tensor position_weights;
    torch::Tensor token_type_weights;

    torch::Tensor layernorm_weight;
    torch::Tensor layernorm_bias;
};
