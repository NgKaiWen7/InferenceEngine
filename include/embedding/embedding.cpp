#include "embedding/embedding.hpp"
#include "safetensors.hpp"
#include <stdfloat>
#include <cstdint>
#include <bit>

void Embedding::load(const std::string file_path)
{
    tensor_loader.load(file_path);
    Tensor embedding_tensor = tensor_loader.get_tensor("embeddings.word_embeddings.weight");
    embedding_dim = embedding_tensor.shape[1];
    vocab_size = embedding_tensor.shape[0];
    embedding_weights = reinterpret_cast<uint16_t *>(embedding_tensor.data);

    Tensor position_tensor = tensor_loader.get_tensor("embeddings.position_embeddings.weight");
    position_embedding_dim = position_tensor.shape[1];
    position_context_size = position_tensor.shape[0];
    position_weights = reinterpret_cast<uint16_t *>(position_tensor.data);

    Tensor token_type_tensor = tensor_loader.get_tensor("embeddings.token_type_embeddings.weight");
    token_type_dim = token_type_tensor.shape[1];
    token_type_size = token_type_tensor.shape[0];
    token_type_weights = reinterpret_cast<uint16_t *>(token_type_tensor.data);

    Tensor bias_tensor = tensor_loader.get_tensor("embeddings.LayerNorm.bias");
    layernorm_bias = reinterpret_cast<uint16_t *>(bias_tensor.data);

    Tensor weight_tensor = tensor_loader.get_tensor("embeddings.LayerNorm.weight");
    layernorm_weight = reinterpret_cast<uint16_t *>(weight_tensor.data);
}

void Embedding::encode(
    const std::vector<int>& token_ids,
    std::vector<std::vector<float>>& embedding_vector)
{
    constexpr float eps = 1e-5f;

    for (size_t i = 0; i < token_ids.size(); ++i)
    {
        int token_id = token_ids[i];
        int position_id = i + 2;

        std::vector<float> embeddings(embedding_dim);

        const uint16_t* word = embedding_weights + token_id * embedding_dim;
        const uint16_t* position = position_weights + position_id * embedding_dim;

        float mean = 0.0f;

        for (size_t j = 0; j < embedding_dim; ++j)
        {
            float w = static_cast<float>(
                std::bit_cast<std::float16_t>(word[j])
            );

            float p = static_cast<float>(
                std::bit_cast<std::float16_t>(position[j])
            );

            float t = static_cast<float>(
                std::bit_cast<std::float16_t>(token_type_weights[j])
            );

            embeddings[j] = w + p + t;
            mean += embeddings[j];
        }

        mean /= embedding_dim;

        float variance = 0.0f;

        for (size_t j = 0; j < embedding_dim; ++j)
        {
            float diff = embeddings[j] - mean;
            variance += diff * diff;
        }

        variance /= embedding_dim;

        float inv_std = 1.0f / std::sqrt(variance + eps);

        for (size_t j = 0; j < embedding_dim; ++j)
        {
            float gamma = static_cast<float>(
                std::bit_cast<std::float16_t>(layernorm_weight[j])
            );

            float beta = static_cast<float>(
                std::bit_cast<std::float16_t>(layernorm_bias[j])
            );

            embeddings[j] =
                (embeddings[j] - mean) * inv_std * gamma + beta;
        }

        embedding_vector.push_back(std::move(embeddings));
    }
}

void Embedding::get_embedding(int token_id, std::vector<std::float16_t> &output)
{
    output.resize(embedding_dim);
    int offset = token_id * embedding_dim;
    for (int i = 0; i < embedding_dim; i++)
    {
        output[i] = std::bit_cast<std::float16_t>(embedding_weights[offset + i]);
    }
}