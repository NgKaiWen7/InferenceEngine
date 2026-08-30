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

auto ids = tokenizer.encode(
    "This is a longer sample sentence for benchmarking the BGE-M3 inference engine. "
    "The purpose is to evaluate the performance of tokenisation, embedding generation, "
    "transformer computation, attention, feed forward layers, layer normalisation, "
    "and memory access patterns under a more realistic input sequence. "
    "We want to compare the performance of the baseline C++ implementation against "
    "an implementation optimised with OpenBLAS and CPU SIMD instructions. "
    "The input should contain enough tokens to make computational differences measurable "
    "while remaining representative of typical semantic embedding workloads."
);
    for (int i : ids)
    {
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
    double total = 0.0;

    for (int i = 0; i < 100; ++i)
    {
        Tensor embedding_vector;
        auto start = std::chrono::high_resolution_clock::now();
        embedding.encode(ids, embedding_vector);
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::micro>(end - start).count();
    }
    std::cout << "Average: " << total / 100.0 << " us\n";
    // std::vector<std::vector<float>> output = embedding_vector;
    // for (int layer = 0; layer < 24; ++layer)
    // {
    //     TransformerLayer transformer;

    //     transformer.load(
    //         "bge-m3-safetensors/model.safetensors",
    //         layer);

    //     std::vector<std::vector<float>> next_output;

    //     transformer.attention(output, next_output);

    //     output = std::move(next_output);
    // }
    // // Mean pooling
    // std::vector<float> embedding_output = output[0];

    // float norm = 0.0f;

    // for (float x : embedding_output)
    //     norm += x * x;

    // norm = std::sqrt(norm);

    // for (float &x : embedding_output)
    //     x /= norm;

    // for (int j = 0; j < 3; j++)
    // {
    //     std::cout << embedding_output[j] << ", ";
    // }
    // std::cout << std::endl;
    // for (int j = 1023; j > 1020; j--)
    // {
    //     std::cout << embedding_output[j] << ", ";
    // }
    // std::cout << std::endl;

    // Pooler pooler;
    // pooler.load("bge-m3-safetensors/model.safetensors");
    // std::vector<std::vector<float>> final_output;
    // std::vector<std::vector<float>> pooler_input = {output[0]};
    // pooler.pool(pooler_input, final_output);
}