#include "embedding/embedding.hpp"
#include "safetensors.hpp"
#include <bit>
#include <cstdint>
#include <immintrin.h>
#include <stdfloat>
#include <torch/torch.h>

void Embedding::load(const std::string file_path) {
    tensor_loader.load(file_path);

    Tensor temp;

    temp = tensor_loader.get_tensor("embeddings.word_embeddings.weight");
    embedding_weights = torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
    embedding_dim = temp.shape[1];

    temp = tensor_loader.get_tensor("embeddings.position_embeddings.weight");
    position_weights = torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor("embeddings.token_type_embeddings.weight");
    token_type_weights = torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor("embeddings.LayerNorm.weight");
    layernorm_weight = torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor("embeddings.LayerNorm.bias");
    layernorm_bias = torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
}

void Embedding::encode(const std::vector<int> &token_ids, torch::Tensor &embeddings) {
    constexpr float eps = 1e-5f;
    embeddings = torch::empty({static_cast<int64_t>(token_ids.size()), embedding_dim}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int64_t token_id = token_ids[i];
        int64_t position_id = i + 2;
        torch::Tensor word = embedding_weights[token_id];
        torch::Tensor position = position_weights[position_id];
        torch::Tensor embedding = word + position + token_type_weights[0];
        torch::Tensor mean = embedding.mean();
        torch::Tensor variance = ((embedding - mean) * (embedding - mean)).mean();
        torch::Tensor inv_std = torch::rsqrt(variance + eps);
        embeddings[i] = (embedding - mean) * inv_std * layernorm_weight + layernorm_bias;
    }
}
