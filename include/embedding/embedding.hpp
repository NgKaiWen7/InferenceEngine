#pragma once

#include <string>
#include <vector>
#include "safetensors.hpp"
#include <stdfloat>

class Embedding
{
public:
    void load(const std::string file_path);
    void encode(const std::vector<int> &token_ids, std::vector<std::vector<std::float16_t>> &embedding_vector);
    void get_embedding(const int token_id, std::vector<std::float16_t> &output);

private:
    int embedding_dim;
    int vocab_size;

    int position_embedding_dim;
    int position_context_size;
    SafeTensorLoader tensor_loader;
    uint16_t *embedding_weights;
    uint16_t *position_weights;
};