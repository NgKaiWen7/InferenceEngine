<<<<<<< HEAD
#include <iostream>
#include "tokenizer/BPE.hpp"
#include "embedding/embedding.hpp"
#include "attention/self_attention.hpp"
#include "utils/immitrin.hpp"
#include <stdfloat>
=======
#include "attention/self_attention.hpp"
#include "embedding/embedding.hpp"
#include "tokenizer/BGEtokenizer.hpp"
#include <cblas.h>
#include <chrono>
>>>>>>> libtorch
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <string>
#include <torch/torch.h>

<<<<<<< HEAD
int main()
{
    BPETokenizer tokenizer;

    if (!tokenizer.load("Qwen3-1.7B-tokenizer/tokenizer.json"))
    {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }
=======
int main() {
    BGEtokenizer tokenizer;
    if (!tokenizer.load("/home/nkw/InferenceEngine/bge-m3-safetensors/sentencepiece.bpe.model")) {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }
    std::ifstream file("/home/nkw/InferenceEngine/text.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    // 2. Read the file buffer into a stringstream
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string file_contents = buffer.str();
    std::replace(file_contents.begin(), file_contents.end(), '\n', ' ');
    std::vector<int> token_ids = tokenizer.encode(file_contents);

    Embedding embedding_layer;
    embedding_layer.load("/home/nkw/InferenceEngine/bge-m3-safetensors/model.safetensors");
    torch::Tensor embeddings;
    embedding_layer.encode(token_ids, embeddings);

    TransformerWorkspace workspace = TransformerWorkspace(token_ids.size());
    std::vector<TransformerLayer> layers(24);
    for (int i = 0; i < 24; ++i)
        layers[i].load("/home/nkw/InferenceEngine/bge-m3-safetensors/model.safetensors", i);

    torch::Tensor input = embeddings;
    input = input.to(torch::kCUDA);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 24; ++i) {
        input = layers[i].forward(input, workspace);
    }
    torch::Tensor final_output = input;
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "24 layers: " << elapsed_ms << " ms\n";
    return 0;
>>>>>>> libtorch
}
