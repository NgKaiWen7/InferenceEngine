#include "embedding/embedding.hpp"
#include "safetensors.hpp"
#include "utils/conversion.hpp"
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

    tensor_loader.load(file_path);
    Tensor position_tensor = tensor_loader.get_tensor("embeddings.position_embeddings.weight");

    position_embedding_dim = position_tensor.shape[1];
    position_context_size = position_tensor.shape[0];
    position_weights = reinterpret_cast<uint16_t *>(position_tensor.data);
}

void Embedding::encode(const std::vector<int> &token_ids, std::vector<std::vector<std::float16_t>> &embedding_vector)
{
    for (size_t i = 0; i < token_ids.size(); ++i)
    {
        int token_id = token_ids[i];
        int position_id = i + 2;

        std::vector<std::float16_t> embeddings(embedding_dim);

        const uint16_t *word = embedding_weights + token_id * embedding_dim;
        const uint16_t *position = position_weights + position_id * position_embedding_dim;

        for (size_t j = 0; j < embedding_dim; ++j)
            embeddings[j] = std::bit_cast<std::float16_t>(word[j]) + std::bit_cast<std::float16_t>(position[j]);
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