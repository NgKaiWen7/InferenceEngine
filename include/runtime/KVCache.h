#pragma once

#include "../tensor/Tensor.h"

struct LayerKV
{
    Tensor key;

    Tensor value;
};

class KVCache
{
public:

    std::vector<LayerKV> layers;
};