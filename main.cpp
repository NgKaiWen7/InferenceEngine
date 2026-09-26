#include <iostream>
#include "tokenizer/BPE.hpp"
#include "embedding/embedding.hpp"
#include "attention/self_attention.hpp"
#include "utils/immitrin.hpp"
#include <stdfloat>
#include <cstdlib>
#include <cblas.h>

int main()
{
    BPETokenizer tokenizer;

    if (!tokenizer.load("Qwen3-1.7B-tokenizer/tokenizer.json"))
    {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }
}
