#pragma once

#include <vector>

#include "ModelConfig.h"
#include "TransformerLayer.h"

class QwenModel
{
public:

    bool load(const std::string& directory);

    std::vector<float> forward(
        const std::vector<int>& tokens);

private:

    ModelConfig config_;

    std::vector<TransformerLayer> layers_;
};