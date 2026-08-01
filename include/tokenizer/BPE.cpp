#include "BPE.hpp"

#include <algorithm>
#include <sstream>
#include <climits>
#include <iostream>
#include <fstream>
#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <nlohmann/json.hpp>

int BPETokenizer::load(const std::string file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open tokenizer.json\n";
        return -1;
    }
    nlohmann::json metadata;
    file >> metadata;
    for (auto &[key, value] : metadata["model"]["vocab"].items())
    {
        vocabulary[key] = value;
    }

    std::vector<std::pair<std::string, std::string>> merges;
    auto &json_merges = metadata["model"]["merges"];
    merges.reserve(json_merges.size());
    for (const auto &value : json_merges)
    {
        std::string first = value[0].get<std::string>();
        std::string second = value[1].get<std::string>();
        merges.emplace_back(first, second);
    }

    for (int i = 0; i < merges.size(); i++)
    {
        merge_rank[merges[i]] = i;
    }
    return 0;
}

std::vector<int> BPETokenizer::encode(
    const std::string &text)
{
    std::vector<int> ids;

    std::vector<std::string> words = pre_tokenize(text);

    for (const std::string &word : words)
    {
        std::string bytes = byte_encode(word);
        auto tokens = bpe(bytes);

        for (const auto &token : tokens)
        {
            auto it = vocabulary.find(token);

            if (it != vocabulary.end())
            {
                ids.push_back(it->second);
            }
            else
            {
                std::cerr << "Unknown token: " << token << std::endl;
            }
        }
    }
    return ids;
}

std::string BPETokenizer::normalize(
    const std::string &text)
{
    return text;
}

std::vector<std::string> BPETokenizer::pre_tokenize(
    const std::string &text)
{
    std::vector<std::string> tokens;

    UErrorCode status = U_ZERO_ERROR;

    icu::UnicodeString pattern =
        UNICODE_STRING_SIMPLE(
            R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)");

    icu::RegexPattern *regex = icu::RegexPattern::compile(pattern, 0, status);

    icu::UnicodeString input = icu::UnicodeString::fromUTF8(text);

    icu::RegexMatcher *matcher = regex->matcher(input, status);

    while (matcher->find(status))
    {
        icu::UnicodeString match = matcher->group(status);

        std::string utf8;

        match.toUTF8String(utf8);

        tokens.push_back(utf8);
    }

    delete matcher;
    delete regex;
    return tokens;
}

std::string BPETokenizer::byte_encode(const std::string &token)
{
    std::string result;
    result = unicode_encoder.encode(token);
    return result;
}

std::vector<std::string> split_utf8(const std::string &input)
{
    std::vector<std::string> result;
    size_t i = 0;
    while (i < input.size())
    {
        size_t len = 1;
        unsigned char c = input[i];
        if ((c & 0x80) == 0)
        {
            len = 1;
        }
        else if ((c & 0xE0) == 0xC0)
        {
            len = 2;
        }
        else if ((c & 0xF0) == 0xE0)
        {
            len = 3;
        }
        else if ((c & 0xF8) == 0xF0)
        {
            len = 4;
        }
        result.push_back(input.substr(i, len));
        i += len;
    }
    return result;
}

std::vector<std::string> BPETokenizer::bpe(const std::string &word)
{
    auto pieces = split_utf8(word);
    int best_rank = INT_MAX;
    while (pieces.size() > 1)
    {
        int best_pos = -1;

        for (size_t i = 0; i + 1 < pieces.size(); ++i)
        {
            auto it = merge_rank.find({pieces[i], pieces[i + 1]});
            if (it != merge_rank.end() &&
                it->second < best_rank)
            {
                best_rank = it->second;
                best_pos = i;
            }
        }

        if (best_pos == -1)
            break;
        pieces[best_pos] += pieces[best_pos + 1];
        pieces.erase(pieces.begin() + best_pos + 1);
    };
    return pieces;
}