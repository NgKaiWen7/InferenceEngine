#include "attention/self_attention.hpp"
#include "safetensors.hpp"
#include "utils/conversion.hpp"
#include <bit>
#include <cblas.h>
#include <cstdint>
#include <stdfloat>

void TransformerLayer::load(const std::string &file_path, int layer) {
    tensor_loader.load(file_path);

    std::string prefix = "encoder.layer." + std::to_string(layer) + ".";

    attention_query_weight = tensor_loader.get_tensor(prefix + "attention.self.query.weight");
    attention_query_bias = tensor_loader.get_tensor(prefix + "attention.self.query.bias");

    attention_key_weight = tensor_loader.get_tensor(prefix + "attention.self.key.weight");
    attention_key_bias = tensor_loader.get_tensor(prefix + "attention.self.key.bias");

    attention_value_weight = tensor_loader.get_tensor(prefix + "attention.self.value.weight");
    attention_value_bias = tensor_loader.get_tensor(prefix + "attention.self.value.bias");

    attention_output_weight = tensor_loader.get_tensor(prefix + "attention.output.dense.weight");
    attention_output_bias = tensor_loader.get_tensor(prefix + "attention.output.dense.bias");

    attention_layernorm_weight = tensor_loader.get_tensor(prefix + "attention.output.LayerNorm.weight");
    attention_layernorm_bias = tensor_loader.get_tensor(prefix + "attention.output.LayerNorm.bias");

    intermediate_weight = tensor_loader.get_tensor(prefix + "intermediate.dense.weight");
    intermediate_bias = tensor_loader.get_tensor(prefix + "intermediate.dense.bias");

    output_weight = tensor_loader.get_tensor(prefix + "output.dense.weight");
    output_bias = tensor_loader.get_tensor(prefix + "output.dense.bias");

    output_layernorm_weight = tensor_loader.get_tensor(prefix + "output.LayerNorm.weight");
    output_layernorm_bias = tensor_loader.get_tensor(prefix + "output.LayerNorm.bias");
}
