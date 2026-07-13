#pragma once

#include <unordered_map>
#include <string>

#include "../tensor/Tensor.h"

class SafeTensorLoader
{
public:

    bool load(const std::string& directory);

    const Tensor& get(const std::string& name) const;

private:

    std::unordered_map<std::string, Tensor> tensors_;
};