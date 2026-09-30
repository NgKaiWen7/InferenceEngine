#include "attention/self_attention.hpp"
#include "safetensors.hpp"
#include "utils/conversion.hpp"
#include <bit>
#include <cblas.h>
#include <cstdint>
#include <stdfloat>
#include <torch/torch.h>

void TransformerLayer::load(const std::string &file_path, int layer) {
    tensor_loader.load(file_path);

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    const std::string prefix = "encoder.layer." + std::to_string(layer) + ".";

    auto load_tensor = [&](const std::string &name) {
        Tensor temp = tensor_loader.get_tensor(prefix + name);
        return torch::from_blob(temp.data, temp.shape, options).clone();
    };

    attention_query_weight = load_tensor("attention.self.query.weight").transpose(0, 1);
    attention_query_bias = load_tensor("attention.self.query.bias");

    attention_key_weight = load_tensor("attention.self.key.weight").transpose(0, 1);
    attention_key_bias = load_tensor("attention.self.key.bias");

    attention_value_weight = load_tensor("attention.self.value.weight").transpose(0, 1);
    attention_value_bias = load_tensor("attention.self.value.bias");

    attention_output_weight = load_tensor("attention.output.dense.weight").transpose(0, 1);
    attention_output_bias = load_tensor("attention.output.dense.bias");

    attention_layernorm_weight = load_tensor("attention.output.LayerNorm.weight");
    attention_layernorm_bias = load_tensor("attention.output.LayerNorm.bias");

    intermediate_weight = load_tensor("intermediate.dense.weight").transpose(0, 1);
    intermediate_bias = load_tensor("intermediate.dense.bias");

    output_weight = load_tensor("output.dense.weight").transpose(0, 1);
    output_bias = load_tensor("output.dense.bias");

    output_layernorm_weight = load_tensor("output.LayerNorm.weight");
    output_layernorm_bias = load_tensor("output.LayerNorm.bias");

    attention_norm_options.weight(attention_layernorm_weight).bias(attention_layernorm_bias).eps(1e-5);

    output_norm_options.weight(output_layernorm_weight).bias(output_layernorm_bias).eps(1e-5);
}
torch::Tensor TransformerLayer::forward(const torch::Tensor &input, TransformerWorkspace &workspace) {
    int64_t sequence_length = input.size(0);

    torch::Tensor &query = workspace.query;
    torch::Tensor &value = workspace.value;
    torch::Tensor &key = workspace.key;

    query = torch::matmul(input, attention_query_weight) + attention_query_bias;
    key = torch::matmul(input, attention_key_weight) + attention_key_bias;
    value = torch::matmul(input, attention_value_weight) + attention_value_bias;

    query = query.view({1, sequence_length, 16, 64}).transpose(1, 2);
    key = key.view({1, sequence_length, 16, 64}).transpose(1, 2);
    value = value.view({1, sequence_length, 16, 64}).transpose(1, 2);

    torch::Tensor &context = workspace.context;
    context = at::scaled_dot_product_attention(query, key, value, c10::nullopt, 0.0, false);
    context = context.transpose(1, 2).contiguous().view({sequence_length, 1024});

    torch::Tensor &attention_dense = workspace.attention_dense;
    attention_dense = torch::matmul(context, attention_output_weight) + attention_output_bias;

    attention_dense = attention_dense + input;
    attention_dense = torch::nn::functional::layer_norm(attention_dense, attention_norm_options);

    torch::Tensor &intermediate = workspace.intermediate;
    intermediate = torch::matmul(attention_dense, intermediate_weight) + intermediate_bias;

    intermediate = torch::gelu(intermediate);
    intermediate = torch::matmul(intermediate, output_weight) + output_bias;

    intermediate = intermediate + attention_dense;

    return torch::nn::functional::layer_norm(intermediate, output_norm_options);
}
