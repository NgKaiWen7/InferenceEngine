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
    Tensor temp;

    std::string prefix = "encoder.layer." + std::to_string(layer) + ".";

    temp = tensor_loader.get_tensor(prefix + "attention.self.query.weight");
    attention_query_weight =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.self.query.bias");
    attention_query_bias =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.self.key.weight");
    attention_key_weight =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.self.key.bias");
    attention_key_bias =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.self.value.weight");
    attention_value_weight =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.self.value.bias");
    attention_value_bias =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.output.dense.weight");
    attention_output_weight =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.output.dense.bias");
    attention_output_bias =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.output.LayerNorm.weight");
    attention_layernorm_weight =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "attention.output.LayerNorm.bias");
    attention_layernorm_bias =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "intermediate.dense.weight");
    intermediate_weight =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "intermediate.dense.bias");
    intermediate_bias =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "output.dense.weight");
    output_weight =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "output.dense.bias");
    output_bias =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "output.LayerNorm.weight");
    output_layernorm_weight =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    temp = tensor_loader.get_tensor(prefix + "output.LayerNorm.bias");
    output_layernorm_bias =
        torch::from_blob(temp.data, temp.shape, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

    std::vector<int64_t> normalized_shape = {1024};
    attention_norm_options.weight(attention_layernorm_weight).bias(attention_layernorm_bias).eps(1e-5);
    output_norm_options.weight(output_layernorm_weight).bias(output_layernorm_bias).eps(1e-5);
}

void TransformerLayer::forward(const torch::Tensor &input, torch::Tensor &output, TransformerWorkspace &workspace) {
    int64_t sequence_length = input.size(0);
    // Input
    torch::Tensor &query = workspace.query;
    torch::Tensor &value = workspace.value;
    torch::Tensor &key = workspace.key;

    // QKV projection
    query = torch::matmul(input, attention_query_weight.transpose(0, 1)) + attention_query_bias;
    key = torch::matmul(input, attention_key_weight.transpose(0, 1)) + attention_key_bias;
    value = torch::matmul(input, attention_value_weight.transpose(0, 1)) + attention_value_bias;

    // Split into heads
    query = query.view({sequence_length, 16, 64});
    query = query.transpose(0, 1);
    key = key.view({sequence_length, 16, 64});
    key = key.transpose(0, 1);
    value = value.view({sequence_length, 16, 64});
    value = value.transpose(0, 1);

    // Attention scores
    torch::Tensor key_transposed = key.transpose(-2, -1);
    torch::Tensor scores = torch::matmul(query, key_transposed);
    scores = scores / std::sqrt(64.0f);
    scores = torch::softmax(scores, -1);

    // Attention × V
    torch::Tensor context = torch::matmul(scores, value);
    context = context.transpose(0, 1);
    context = context.contiguous();
    context = context.view({sequence_length, 1024});

    // Attention output projection
    torch::Tensor &attention_dense = workspace.attention_dense;
    attention_dense = torch::matmul(context, attention_output_weight.transpose(0, 1)) + attention_output_bias;

    // Residual
    attention_dense = attention_dense + input;

    // Attention LayerNorm
    attention_dense = torch::nn::functional::layer_norm(attention_dense, attention_norm_options);

    // FFN
    torch::Tensor &intermediate = workspace.intermediate;
    intermediate = torch::matmul(attention_dense, intermediate_weight.transpose(0, 1)) + intermediate_bias;

    // GELU
    intermediate = 0.5f * intermediate * (1.0f + torch::erf(intermediate * 0.7071067811865475f));

    // FFN output projection
    output = torch::matmul(intermediate, output_weight.transpose(0, 1)) + output_bias;

    // Residual
    output = output + attention_dense;

    // Final LayerNorm
    output = torch::nn::functional::layer_norm(output, output_norm_options);
}
