#pragma once

#include <string>
#include <vector>
#include "safetensors.hpp"
#include <stdfloat>

class Pooler
{
public:
    void load(const std::string file_path);
    void pool(const std::vector<std::vector<float>> &encoder_output, std::vector<std::vector<float>> &output);
    void linear(
        const std::vector<float> &input,
        const uint16_t *weight,
        const uint16_t *bias,
        int input_size,
        int output_size,
        std::vector<float> &output);

private:
    SafeTensorLoader tensor_loader;
    uint16_t *pooler_weights;
    uint16_t *pooler_bias;
};