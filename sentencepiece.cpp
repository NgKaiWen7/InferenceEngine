#include <iostream>
#include <vector>
#include <sentencepiece_processor.h>

int main()
{
    sentencepiece::SentencePieceProcessor sp;

    auto status = sp.Load("bge-m3-safetensors/sentencepiece.bpe.model");

    if (!status.ok()) {
        std::cerr << status.ToString() << '\n';
        return 1;
    }

    std::string text = "hello world";

    std::vector<std::string> pieces;
    std::vector<int> ids;

    sp.Encode(text, &pieces);
    sp.Encode(text, &ids);

    std::cout << "Pieces:\n";
    for (const auto& piece : pieces)
        std::cout << piece << '\n';

    std::cout << "\nIDs:\n";
    for (int id : ids)
        std::cout << id << ' ';

    std::cout << '\n';
}