#pragma once

#include <vector>

class Tensor
{
public:

    Tensor() = default;

    Tensor(std::vector<int> shape);

    float* data();

    const float* data() const;

    int rows() const;

    int cols() const;

    const std::vector<int>& shape() const;

private:

    std::vector<int> shape_;

    std::vector<float> data_;
};