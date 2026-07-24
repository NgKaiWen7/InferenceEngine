#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

int main()
{
    std::ifstream file("models/qwen3-4B/vocab.json");
    if (!file.is_open())
    {
        std::cerr << "Failed to open config.json\n";
        return 1;
    }

    nlohmann::json metadata;
    file >> metadata;

    std::cout << metadata.dump(4) << std::endl;

    return 0;
}