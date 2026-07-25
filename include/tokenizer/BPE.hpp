#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <tokenizer/UnicodeEncoder.hpp>

struct PairHash
{
    size_t operator()(
        const std::pair<std::string,std::string>& p
    ) const
    {
        return std::hash<std::string>()(p.first) ^
              (std::hash<std::string>()(p.second) << 1);
    }
};


class BPETokenizer
{

public:

    void load(
        const std::unordered_map<std::string,int>& vocab,
        const std::vector<std::pair<std::string,std::string>>& merges
    );


    std::vector<int> encode(
        const std::string& text
    );


private:

    std::string normalize(
        const std::string& text
    );


    std::vector<std::string> pre_tokenize(
        const std::string& text
    );


    std::string byte_encode(
        const std::string& token
    );


    std::vector<std::string> bpe(
        const std::string& pieces
    );

    std::unordered_map<std::string,int> vocab_;
    UnicodeEncoder unicode_encoder;


    std::unordered_map<
        std::pair<std::string,std::string>,
        int,
        PairHash
    > merge_rank_;

};