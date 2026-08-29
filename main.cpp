#include <iostream>
#include "tokenizer/BGEtokenizer.hpp"
#include "embedding/embedding.hpp"
#include "attention/self_attention.hpp"
#include <stdfloat>

int main()
{
    BGEtokenizer tokenizer;

    if (!tokenizer.load("bge-m3-safetensors/sentencepiece.bpe.model"))
    {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }

    auto ids = tokenizer.encode("b");
    for (int i: ids){
        std::cout << i << ", ";
    }
    std::cout << std::endl;
    Embedding embedding;
    embedding.load("bge-m3-safetensors/model.safetensors");
    // std::vector<std::float16_t> vector;
    // embedding.get_embedding(0, vector);
    //     std::cout << "vector size: " << vector.size() << '\n';
    //     for (int i = 0; i < 10; i++){
    //         std::cout << vector[i] << ", ";
    //     }
    //     std::cout << std::endl;
    std::vector<std::vector<float>> embedding_vector;
    embedding.encode(ids, embedding_vector);
    // for (const auto& vector: embedding_vector){
    //     std::cout << "embedding_vector size: " << vector.size() << '\n';
    //     for (int i = 0; i < 10; i++){
    //         std::cout << vector[i] << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    TransformerLayer transformer;
    transformer.load("bge-m3-safetensors/model.safetensors", 0);
    std::vector<std::vector<float>> output;
    transformer.attention(embedding_vector, output);

    TransformerLayer transformer2;
    transformer.load("bge-m3-safetensors/model.safetensors", 1);
    std::vector<std::vector<float>> output2;
    transformer.attention(output, output2);
    for (size_t i = 0; i < 3; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            std::cout << output2[i][j] << ", ";
        }
        std::cout << std::endl;
    }
}