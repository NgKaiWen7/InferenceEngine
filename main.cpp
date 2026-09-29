#include "attention/self_attention.hpp"
#include "embedding/embedding.hpp"
#include "tokenizer/BGEtokenizer.hpp"
#include <cblas.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdfloat>
#include <torch/torch.h>
#include <utility>

int main() {
    BGEtokenizer tokenizer;
    std::cout << torch::show_config() << std::endl;
    if (!tokenizer.load("bge-m3-safetensors/sentencepiece.bpe.model")) {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }

    Embedding embedding_layer;
    std::vector<int> token_ids(4096, 1);
    embedding_layer.load("bge-m3-safetensors/model.safetensors");
    torch::Tensor embeddings;
    embedding_layer.encode(token_ids, embeddings);

    TransformerWorkspace workspace = TransformerWorkspace(token_ids.size());
    std::vector<TransformerLayer> layers(24);
    for (int i = 0; i < 24; ++i)
        layers[i].load("bge-m3-safetensors/model.safetensors", i);

    torch::Tensor input = embeddings;
    torch::Tensor output = embeddings;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 24; ++i) {
        layers[i].forward(input, output, workspace);
        input = output;
    }
    torch::Tensor final_output = input;
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "24 layers: " << elapsed_ms << " ms\n";
    return 0;
}
