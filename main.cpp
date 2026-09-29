#include "embedding/embedding.hpp"
#include "tokenizer/BGEtokenizer.hpp"
#include <cblas.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdfloat>

int main() {
    BGEtokenizer tokenizer;

    if (!tokenizer.load("bge-m3-safetensors/sentencepiece.bpe.model")) {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }

    Embedding embedding_layer;
    std::vector<int> token_ids(4096, 1);
    embedding_layer.load("bge-m3-safetensors/model.safetensors");
    torch::Tensor embeddings;
    auto start = std::chrono::steady_clock::now();
    embedding_layer.encode(token_ids, embeddings);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "encode: " << elapsed.count() << " ms\n";
}
