#include <iostream>
#include "tokenizer/BGEtokenizer.hpp"
#include "embedding/embedding.hpp"
#include "attention/self_attention.hpp"
#include "pooler/pooler.hpp"
#include <stdfloat>
#include <chrono>

int main()
{
    BGEtokenizer tokenizer;
    
    if (!tokenizer.load("bge-m3-safetensors/sentencepiece.bpe.model"))
    {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }
    
    Embedding embedding;
    embedding.load("bge-m3-safetensors/model.safetensors");
    
    std::vector<TransformerLayer> layers(24);
    for (int i = 0; i < 24; ++i)
    layers[i].load("bge-m3-safetensors/model.safetensors", i);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto token_ids = tokenizer.encode(
        "This is a longer sample sentence for benchmarking the BGE-M3 inference engine. "
        "The purpose is to evaluate the performance of tokenisation, embedding generation, "
        "transformer computation, attention, feed forward layers, layer normalisation, "
        "and memory access patterns under a more realistic input sequence. "
        "We want to compare the performance of the baseline C++ implementation against "
        "an implementation optimised with OpenBLAS and CPU SIMD instructions. "
        "The input should contain enough tokens to make computational differences measurable "
        "while remaining representative of typical semantic embedding workloads.");
        
    Tensor embedding_vector;
    embedding_vector.size = token_ids.size() * 1024;
    embedding_vector.shape = {static_cast<int64_t>(token_ids.size()), static_cast<int64_t>(1024)};
    embedding_vector.data = new float[embedding_vector.size];

    embedding.encode(token_ids, embedding_vector);
    for (int layer = 0; layer < 24; ++layer)
    {
        Tensor next_output;
        layers[layer].attention(embedding_vector, next_output);
        embedding_vector = next_output;
    }
    float *embedding_output = embedding_vector.data;

    float norm = 0.0f;

    for (size_t j = 0; j < 1024; ++j)
    {
        norm += embedding_output[j] * embedding_output[j];
    }

    norm = std::sqrt(norm);

    for (size_t j = 0; j < 1024; ++j)
        embedding_output[j] /= norm;

    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Average: " << total << " ms\n";
}