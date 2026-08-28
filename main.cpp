#include <iostream>
#include "tokenizer/BGEtokenizer.hpp"
#include "embedding/embedding.hpp"

int main()
{
    BGEtokenizer tokenizer;

    if (!tokenizer.load("xlm-roberta-base/sentencepiece.bpe.model"))
    {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }

    // auto ids = tokenizer.encode("h");

    // for (int id : ids)
    //     std::cout << id << ' ';

    // std::cout << '\n';

    Embedding embedding;
    embedding.load("bge-m3-safetensors/model.safetensors");
    std::vector<std::vector<float>> embedding_vector;
    std::vector<int> ids = {1096};
    embedding.encode(ids, embedding_vector);
    for (std::vector<float> vector: embedding_vector){
        for (float emd: vector){
            std::cout << emd << ", ";
        }
        std::cout << std::endl;
    }

}