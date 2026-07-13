#pragma once

#include <vector>

class Sampler
{
public:

    int greedy(const std::vector<float>& logits);

    int topK(const std::vector<float>& logits,
             int k,
             float temperature);

    int topP(const std::vector<float>& logits,
             float p,
             float temperature);
};