#include "embedding/embedding.hpp"
#include "safetensors.hpp"
#include <stdfloat>
#include <cstdint>
#include <bit>
#include <immintrin.h>

void Embedding::load(const std::string file_path)
{
    tensor_loader.load(file_path);
    embedding_weights = tensor_loader.get_tensor("embeddings.word_embeddings.weight");
    embedding_dim = embedding_weights.shape[1];
    position_weights = tensor_loader.get_tensor("embeddings.position_embeddings.weight");
    token_type_weights = tensor_loader.get_tensor("embeddings.token_type_embeddings.weight");
    layernorm_weight = tensor_loader.get_tensor("embeddings.LayerNorm.bias");
    layernorm_bias = tensor_loader.get_tensor("embeddings.LayerNorm.weight");
}

void Embedding::encode(
    const std::vector<int> &token_ids,
    Tensor embeddings)
{
    constexpr float eps = 1e-5f;
    embeddings.size = token_ids.size() * embedding_dim;
    embeddings.data = new float[embeddings.size];
    #pragma omp parallel for
    for (size_t i = 0; i < token_ids.size(); ++i)
    {
        int token_id = token_ids[i];
        int position_id = i + 2;

        size_t word_idx = token_id * embedding_dim;
        size_t position_idx = position_id * embedding_dim;

        const float *word = embedding_weights.data + word_idx;
        const float *position = position_weights.data + position_idx;
        const float *layernorm_w = layernorm_weight.data;
        const float *layernorm_b = layernorm_bias.data;
        
        int start_idx = i * embedding_dim;
        float *embedding = embeddings.data + start_idx;

        for (size_t j = 0; j < embedding_dim; ++j)
        {
            embedding[j] = word[j] + position[j] + token_type_weights.data[j];
        }

        float mean = 0.0f;
        for (size_t j = 0; j < embedding_dim; ++j)
        {
            mean += embedding[j];
        }
        mean /= embedding_dim;

        float variance = 0.0f;

        for (size_t j = 0; j < embedding_dim; ++j)
        {
            float diff = embedding[j] - mean;
            variance += diff * diff;
        }

        variance /= embedding_dim;

        float inv_std = 1.0f / std::sqrt(variance + eps);

        for (size_t j = 0; j < embedding_dim; ++j)
        {
            float gamma = layernorm_w[j];

            float beta = layernorm_b[j];

            embedding[j] = (embedding[j] - mean) * inv_std * gamma + beta;
        }
    }
}