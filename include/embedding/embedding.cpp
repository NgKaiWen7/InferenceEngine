#include "embedding/embedding.hpp"
#include "safetensors.hpp"
#include "utils/conversion.hpp"

void Embedding::load(const Tensor &tensor)
{
    embedding_dim = tensor.shape[1];
    vocab_size = tensor.shape[0];
    embedding_weights = reinterpret_cast<uint16_t *>(tensor.data);
}

void Embedding::encode(const std::vector<int> &token_ids, std::vector<std::vector<float>> &embedding_vector)
{
    for (const int &row : token_ids)
    {
        std::vector<float> embeddings;
        get_embedding(row, embeddings);
        embedding_vector.push_back(embeddings);
    }
}

void Embedding::get_embedding(int token_id, std::vector<float> &output)
{
    output.resize(embedding_dim);
    size_t offset = token_id * embedding_dim;
    for (int i = 0; i < embedding_dim; i++)
    {
        output[i] = bf16_to_float(embedding_weights[offset + i]);
    }
}