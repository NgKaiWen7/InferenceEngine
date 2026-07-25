#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <tokenizer/BPE.hpp>

int main()
{
    std::string file_path = "models/qwen3-4B/tokenizer.json";

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open tokenizer.json\n";
        return 1;
    }

    nlohmann::json metadata;
    file >> metadata;

    std::unordered_map<std::string, int> vocabulary;
    for (auto& [key, value] : metadata["model"]["vocab"].items())
    {
        vocabulary[key] = value;
    }

    std::vector<std::pair<std::string, std::string>> merges;
    auto& json_merges = metadata["model"]["merges"];
    merges.reserve(json_merges.size());
    for (const auto& value : json_merges)
    {
        std::string first = value[0].get<std::string>();
        std::string second = value[1].get<std::string>();
        merges.emplace_back(first, second);
    }

    BPETokenizer tokenizer;
    tokenizer.load(vocabulary, merges);
    std::vector<int> output = tokenizer.encode("I my name's kai wen!");

    for (int i : output){
        std::cout << i << std::endl;
    }
    return 0;
}