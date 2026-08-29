#include <iostream>
#include "tokenizer/BGEtokenizer.hpp"
#include "embedding/embedding.hpp"
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

    for (int id : ids)
        std::cout << id << ' ';

    std::cout << '\n';

    Embedding embedding;
    embedding.load("bge-m3-safetensors/model.safetensors");
    // std::vector<std::float16_t> vector;
    // embedding.get_embedding(0, vector);
    //     std::cout << "vector size: " << vector.size() << '\n';
    //     for (int i = 0; i < 10; i++){
    //         std::cout << vector[i] << ", ";
    //     }
    //     std::cout << std::endl;
    std::vector<std::vector<std::float16_t>> embedding_vector;
    embedding.encode(ids, embedding_vector);
    for (const auto& vector: embedding_vector){
        std::cout << "embedding_vector size: " << vector.size() << '\n';
        for (int i = 0; i < 10; i++){
            std::cout << vector[i] << ", ";
        }
        std::cout << std::endl;
    }

}