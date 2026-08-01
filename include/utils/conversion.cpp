#include <cstring>
#include <cstdint>
#include "conversion.hpp"

float bf16_to_float(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;

    float result;
    std::memcpy(&result, &bits, sizeof(float));

    return result;
}