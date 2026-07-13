#pragma once

#include <string>

struct ModelConfig
{
    int vocab_size;

    int hidden_size;

    int intermediate_size;

    int num_layers;

    int num_attention_heads;

    int num_key_value_heads;

    int max_position_embeddings;

    float rope_theta;

    float rms_norm_eps;
};