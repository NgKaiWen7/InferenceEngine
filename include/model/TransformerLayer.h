#pragma once

#include "../tensor/Tensor.h"

class TransformerLayer
{
public:

    Tensor forward(const Tensor& input);

private:

    Tensor attention(const Tensor& x);

    Tensor mlp(const Tensor& x);
};