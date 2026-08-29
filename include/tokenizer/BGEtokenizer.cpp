#include "BGEtokenizer.hpp"
#include <iostream>


bool BGEtokenizer::load(const std::string& model_path)
{
    return sp.Load(model_path).ok();
}

std::vector<int> BGEtokenizer::encode(const std::string& text)
{
    std::vector<int> ids;

    auto status = sp.Encode(text, &ids);
    if (!status.ok())
    return {};
    ids.insert(ids.begin(), 1);
    ids.push_back(2);
    for (size_t i = 0; i< ids.size(); i ++){
        ids[i] = convert_id(ids[i]);
    }
    return ids;
}

int BGEtokenizer::convert_id(int id)
{
    switch (id)
    {
        case 0: return 3; // <unk>
        case 1: return 0; // <s>
        case 2: return 2; // </s>
        default: return id + 1;
    }
}