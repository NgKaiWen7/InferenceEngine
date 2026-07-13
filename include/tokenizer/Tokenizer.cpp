#include "Tokenizer.h"

#include <tokenizers_cpp.h>

class TokenizerImpl
{
public:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
};

static TokenizerImpl g_impl;

bool Tokenizer::load(const std::string& tokenizerFile)
{
    g_impl.tokenizer = tokenizers::Tokenizer::FromFile(tokenizerFile);

    return g_impl.tokenizer != nullptr;
}

std::vector<int> Tokenizer::encode(const std::string& text)
{
    std::vector<int> result;

    if (!g_impl.tokenizer)
        return result;

    auto encoding = g_impl.tokenizer->Encode(text);

    const auto& ids = encoding->Ids();

    result.reserve(ids.size());

    for (uint32_t id : ids)
        result.push_back(static_cast<int>(id));

    return result;
}

std::string Tokenizer::decode(const std::vector<int>& tokens)
{
    if (!g_impl.tokenizer)
        return "";

    std::vector<uint32_t> ids;
    ids.reserve(tokens.size());

    for (int token : tokens)
        ids.push_back(static_cast<uint32_t>(token));

    return g_impl.tokenizer->Decode(ids);
}