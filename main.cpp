#include <iostream>
#include <string>
#include "tokenizer/BPE.hpp"
#include "safetensors.hpp"
#include "embedding/embedding.hpp"
#include "attention/casual_attention.hpp"

int main()
{
    /*
    BPETokenizer tokenizer;
    tokenizer.load("models/qwen3-4B/tokenizer.json");
    std::vector<int> output = tokenizer.encode("hellow tokenization");

    for (int i : output){
        std::cout << i << std::endl;
    }
    for (int i : output){
        std::cout << i << std::endl;
    }
    */
    std::string file_path = "models/qwen3-4B/model-00001-of-00003.safetensors";
    Embedding embedding_layer;
    embedding_layer.load(file_path);
    std::vector<std::vector<float>> embeddings;
    std::vector<int> token_ids = {1};
    embedding_layer.encode(token_ids, embeddings);
    
    Attention attention_layer;
    attention_layer.load(file_path);
    std::vector<float> output;
    attention_layer.forward(embeddings, output);

    return 0;
}