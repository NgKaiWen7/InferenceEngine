#include <bit>
#include <cstdint>

float bf16_to_float(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;

    return std::bit_cast<float>(bits);
}