#include <bit>
#include <cstdint>
#include "safetensors.hpp"
#include <stdfloat>

void to_float(const char *src, float *params, size_t size)
{
    const uint16_t *fp16 = reinterpret_cast<const uint16_t *>(src);

    for (size_t i = 0; i < size; ++i)
        params[i] = static_cast<float>(
            std::bit_cast<std::float16_t>(fp16[i])
        );
}