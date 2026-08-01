#pragma once

#include <string>
#include <vector>
#include "safetensors.hpp"

class Embedding
{
public:
    void load(const Tensor &tensor);
    void encode(const std::vector<int> &token_ids, std::vector<std::vector<float>> &embedding_vector);

private:
    void get_embedding(const int token_id, std::vector<float> &output);
    int embedding_dim;
    int vocab_size;
    uint16_t *embedding_weights;
};