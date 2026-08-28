#include "BGEtokenizer.hpp"

bool BGEtokenizer::load(const std::string& model_path)
{
    return sp.Load(model_path).ok();
}

std::vector<int> BGEtokenizer::encode(const std::string& text) const
{
    std::vector<int> ids;

    auto status = sp.Encode(text, &ids);

    if (!status.ok())
        return {};
    return ids;
}