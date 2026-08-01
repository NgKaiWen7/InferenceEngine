#include <iostream>
#include <string>
#include "tokenizer/BPE.hpp"
#include "safetensors.hpp"
#include "embedding/embedding.hpp"


int main()
{
    BPETokenizer tokenizer;
    tokenizer.load("models/qwen3-4B/tokenizer.json");
    std::vector<int> output = tokenizer.encode("hellow tokenization");

    for (int i : output){
        std::cout << i << std::endl;
    }
    /*
    std::string file_path = "models/qwen3-4B/tokenizer.json";

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open tokenizer.json\n";
        return 1;
    }

    nlohmann::json metadata;
    file >> metadata;

    std::unordered_map<std::string, int> vocabulary;
    for (auto& [key, value] : metadata["model"]["vocab"].items())
    {
        vocabulary[key] = value;
    }

    std::vector<std::pair<std::string, std::string>> merges;
    auto& json_merges = metadata["model"]["merges"];
    merges.reserve(json_merges.size());
    for (const auto& value : json_merges)
    {
        std::string first = value[0].get<std::string>();
        std::string second = value[1].get<std::string>();
        merges.emplace_back(first, second);
    }

    BPETokenizer tokenizer;
    tokenizer.load(vocabulary, merges);
    std::vector<int> output = tokenizer.encode("hellow tokenization");

    for (int i : output){
        std::cout << i << std::endl;
    }
    std::string file_path = "models/qwen3-4B/model-00001-of-00003.safetensors";
    SafeTensorLoader tensor_loader;
    tensor_loader.load(file_path);
    Tensor embedding_tensor = tensor_loader.get_tensor("model.embed_tokens.weight");
    
    Embedding embedding_layer;
    embedding_layer.load(embedding_tensor);
    std::vector<std::vector<float>> embeddings = {{}};
    std::vector<int> token_ids = {1, 2, 4};
    embedding_layer.encode(token_ids, embeddings);
    
    for (size_t i = 0; i < embeddings.size(); i++)
    {
        std::cout << "Token " << token_ids[i] << ":\n";

        size_t limit = std::min<size_t>(10, embeddings[i].size());

        for (size_t j = 0; j < limit; j++)
        {
            std::cout << embeddings[i][j] << " ";
        }

        std::cout << "\n\n";
    }
    */

    // std::string file_path = "models/qwen3-4B/model-00001-of-00003.safetensors";
    // SafeTensorLoader tensor_loader;
    // tensor_loader.load(file_path);
    // Tensor input_layernorm = tensor_loader.get_tensor("model.layers.0.input_layernorm.weight");
    // Tensor mlp_proj_down = tensor_loader.get_tensor("model.layers.0.mlp.down_proj.weight");
    // Tensor mlp_gate_proj = tensor_loader.get_tensor("model.layers.0.mlp.gate_proj.weight");
    // Tensor mlp_up_proj = tensor_loader.get_tensor("model.layers.0.mlp.up_proj.weight");
    // Tensor post_attention_layernorm = tensor_loader.get_tensor("model.layers.0.post_attention_layernorm.weight");
    // Tensor k_norm = tensor_loader.get_tensor("model.layers.0.self_attn.k_norm.weight");
    // Tensor k_proj = tensor_loader.get_tensor("model.layers.0.self_attn.k_proj.weight");
    // Tensor o_proj = tensor_loader.get_tensor("model.layers.0.self_attn.o_proj.weight");
    // Tensor q_norm = tensor_loader.get_tensor("model.layers.0.self_attn.q_norm.weight");
    // Tensor q_proj = tensor_loader.get_tensor("model.layers.0.self_attn.q_proj.weight");
    // Tensor v_proj = tensor_loader.get_tensor("model.layers.0.self_attn.v_proj.weight");

    return 0;
}