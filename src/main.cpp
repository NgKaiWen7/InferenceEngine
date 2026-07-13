#include <iostream>

#include "model/QwenModel.h"
#include "tokenizer/Tokenizer.h"

int main()
{
    Tokenizer tokenizer;

    tokenizer.load(
        "models/qwen3-4b/tokenizer.json");

    QwenModel model;

    model.load(
        "models/qwen3-4b");

    auto tokens =
        tokenizer.encode(
            "Hello, who are you?");

    auto logits =
        model.forward(tokens);

    std::cout << "Forward completed.\n";
}