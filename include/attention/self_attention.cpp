#include "attention/self_attention.hpp"
#include "safetensors.hpp"
#include "utils/conversion.hpp"
#include "utils/immitrin.hpp"
#include <stdfloat>
#include <cstdint>
#include <bit>
#include <cblas.h>

void TransformerLayer::load(const std::string &file_path, int layer)
{
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
void TransformerLayer::linear(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    Tensor &output)
{
    size_t M = input.shape[0];
    size_t K = input.shape[1];
    size_t N = weight.shape[0];
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M, N, K,
        1.0f,
        input.data, K,
        weight.data, K,
        0.0f,
        output.data, N);

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            output.data[i * N + j] += bias.data[j];

}

void TransformerLayer::attention(const Tensor &input, Tensor &output, TransformerWorkspace &workspace)
{
    size_t sequence_length = input.shape[0];
    Tensor &query = workspace.query;
    Tensor &value = workspace.value;
    Tensor &key = workspace.key;
    linear(input, attention_value_weight, attention_value_bias, value);
    linear(input, attention_query_weight, attention_query_bias, query);
    linear(input, attention_key_weight, attention_key_bias, key);
    
    Tensor &scores = workspace.scores;
    Tensor &context = workspace.context;
    QKV(query, value, key, num_heads, head_dim, sequence_length, hidden_size,scaling, scores, context);

    Tensor &attention_dense = workspace.attention_dense;
    linear(context, attention_output_weight, attention_output_bias, attention_dense);
    
    residual(attention_dense, input);
    
    layer_norm(attention_dense, attention_layernorm_weight, attention_layernorm_bias);
    
    Tensor &intermediate = workspace.intermediate;
    linear(attention_dense, intermediate_weight, intermediate_bias, intermediate);

    #pragma omp parallel for
    for (size_t i = 0; i < intermediate.size; i++){
        intermediate.data[i] = 0.5f * intermediate.data[i] * (1.0f + std::erf(intermediate.data[i] * 0.7071067811865475f));
    }
    // gelu(intermediate);
    
    linear(intermediate, output_weight, output_bias, output);
    
    residual(output, attention_dense);
    
    layer_norm(output, output_layernorm_weight, output_layernorm_bias);
}