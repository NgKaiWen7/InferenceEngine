#pragma once

#include <string>
#include <vector>
#include <sentencepiece_processor.h>

class BGEtokenizer 
{
private:
    sentencepiece::SentencePieceProcessor sp;
    int convert_id(int id);
public:
    bool load(const std::string& model_path);
    std::vector<int> encode(const std::string& text);
};