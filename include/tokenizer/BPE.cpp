#include "BPE.hpp"

#include <algorithm>
#include <regex>
#include <sstream>
#include <climits>
#include <iostream>

void BPETokenizer::load(
    const std::unordered_map<std::string,int>& vocab,
    const std::vector<std::pair<std::string,std::string>>& merges
)
{
    vocab_ = vocab;
    for(int i=0;i<merges.size();i++)
    {
        merge_rank_[merges[i]] = i;
    }
}

std::vector<int> BPETokenizer::encode(
    const std::string& text
)
{
    std::vector<int> ids;

    auto normalized = normalize(text);

    auto words = pre_tokenize(normalized);


    for (const auto& word : words)
    {
        auto bytes = byte_encode(word);

        auto tokens = bpe(bytes);
        
        for (const auto& token : tokens)
        {
            std::cout << token << std::endl;
            
            auto it = vocab_.find(token);

            if (it != vocab_.end())
            {
                ids.push_back(it->second);
            }
            else
            {
                std::cerr
                    << "Unknown token: "
                    << token
                    << "\n";
            }
        }
    }

    return ids;
}

std::string BPETokenizer::normalize(
    const std::string& text
)
{
    return text;
}

std::vector<std::string> BPETokenizer::pre_tokenize(
    const std::string& text
)
{
    std::vector<std::string> tokens;


    std::regex pattern(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]| ?[^A-Za-z0-9\s]+|\s+)"
    );


    auto begin =
        std::sregex_iterator(
            text.begin(),
            text.end(),
            pattern
        );


    auto end =
        std::sregex_iterator();


    for(auto it = begin; it != end; ++it)
    {
        tokens.push_back(
            it->str()
        );
    }


    return tokens;
}

std::vector<std::string>
BPETokenizer::byte_encode(
    const std::string& token
)
{
    std::vector<std::string> result;


    for(unsigned char c : token)
    {
        if(c == ' ')
        {
            result.push_back("Ġ");
        }
        else
        {
            result.push_back(
                std::string(1,c)
            );
        }
    }


    return result;
}

std::vector<std::string> BPETokenizer::bpe(
    const std::vector<std::string>& input
)
{
    auto pieces = input;


    while(pieces.size() > 1)
    {

        int best_rank = INT_MAX;

        int best_pos = -1;


        for(int i=0;i<pieces.size()-1;i++)
        {

            auto pair =
                std::make_pair(
                    pieces[i],
                    pieces[i+1]
                );


            auto it =
                merge_rank_.find(pair);


            if(it != merge_rank_.end())
            {
                std::cout 
                    << "Found merge: "
                    << pieces[i]
                    << " + "
                    << pieces[i+1]
                    << " rank="
                    << it->second
                    << "\n";
                if(it->second < best_rank)
                {
                    best_rank = it->second;
                    best_pos = i;
                }
            }
        }


        if(best_pos == -1)
            break;


        pieces[best_pos] =
            pieces[best_pos] +
            pieces[best_pos+1];


        pieces.erase(
            pieces.begin()+best_pos+1
        );
    }


    return pieces;
}