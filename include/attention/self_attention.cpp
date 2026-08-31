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

    std::cout
        << "Linear: "
        << "input=[" << M << ", " << K << "] "
        << "weight=[" << weight.shape[0] << ", " << weight.shape[1] << "] "
        << "bias=[" << bias.shape[0] << "] "
        << "output=[" << output.shape[0] << ", " << output.shape[1] << "] "
        << "GEMM=(" << M << "x" << K << ") * ("
        << K << "x" << N << ")"
        << std::endl;

    auto gemm_start = std::chrono::high_resolution_clock::now();

    gemm_avx2(input.data, weight.data, output.data, M, K, N);
    // cblas_sgemm(
    //     CblasRowMajor,
    //     CblasNoTrans,
    //     CblasTrans,
    //     M, N, K,
    //     1.0f,
    //     input.data, K,
    //     weight.data, K,
    //     0.0f,
    //     output.data, N
    // );

    auto gemm_end = std::chrono::high_resolution_clock::now();

    double gemm_ms =
        std::chrono::duration<double, std::milli>(
            gemm_end - gemm_start).count();

    auto bias_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            output.data[i * N + j] += bias.data[j];

    auto bias_end = std::chrono::high_resolution_clock::now();

    double bias_ms =
        std::chrono::duration<double, std::milli>(
            bias_end - bias_start).count();

    std::cout
        << "  GEMM: " << gemm_ms << " ms"
        << " | Bias: " << bias_ms << " ms"
        << std::endl;
}

void TransformerLayer::attention(const Tensor &input, Tensor &output, TransformerWorkspace &workspace)
{
    auto inputstart = std::chrono::high_resolution_clock::now();

    size_t sequence_length = input.shape[0];
    Tensor &query = workspace.query;
    Tensor &value = workspace.value;
    Tensor &key = workspace.key;
    linear(input, attention_value_weight, attention_value_bias, value);
    linear(input, attention_query_weight, attention_query_bias, query);
    linear(input, attention_key_weight, attention_key_bias, key);
    
    auto inputend = std::chrono::high_resolution_clock::now();
    double inputtotal = std::chrono::duration<double, std::milli>(inputend - inputstart).count();
    std::cout << "Input operations: " << inputtotal << " ms\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor &scores = workspace.scores;
    Tensor &context = workspace.context;

    #pragma omp parallel for
    for (int h = 0; h < num_heads; h++)
    {
        int offset = h * head_dim;
        int score_offset = h * sequence_length * sequence_length;

        for (size_t i = 0; i < sequence_length; i++)
        {
            for (size_t j = 0; j < sequence_length; j++)
            {
                float sum = 0.0f;

                for (int k = 0; k < head_dim; k++)
                {
                    sum += query.data[offset + i * hidden_size + k] * key.data[offset + j * hidden_size + k];
                }
                scores.data[score_offset + i * sequence_length + j] = sum / std::sqrt(static_cast<float>(head_dim));
            }
        }

        for (size_t i = 0; i < sequence_length; ++i)
        {
            float *row = scores.data + score_offset + i * sequence_length;

            float max_value = row[0];

            for (size_t j = 1; j < sequence_length; ++j)
                max_value = std::max(max_value, row[j]);

            float sum = 0.0f;

            for (size_t j = 0; j < sequence_length; ++j)
            {
                row[j] = std::exp(row[j] - max_value);
                sum += row[j];
            }

            for (size_t j = 0; j < sequence_length; ++j)
                row[j] /= sum;
        }

        // Attention × V
        for (size_t i = 0; i < sequence_length; i++)
        {
            for (int k = 0; k < head_dim; k++)
            {
                float sum = 0.0f;

                for (size_t j = 0; j < sequence_length; j++)
                    sum += scores.data[score_offset + i * sequence_length + j] * value.data[offset + j * hidden_size + k];
                context.data[offset + i * hidden_size + k] = sum;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Head operations: " << total << " ms\n";
start = std::chrono::high_resolution_clock::now();

Tensor &attention_dense = workspace.attention_dense;
linear(context, attention_output_weight, attention_output_bias, attention_dense);

end = std::chrono::high_resolution_clock::now();
std::cout << "Attention output linear: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms\n";


start = std::chrono::high_resolution_clock::now();

residual(attention_dense, input);

end = std::chrono::high_resolution_clock::now();
std::cout << "Residual 1: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms\n";


start = std::chrono::high_resolution_clock::now();

layer_norm(attention_dense, attention_layernorm_weight, attention_layernorm_bias);

end = std::chrono::high_resolution_clock::now();
std::cout << "LayerNorm 1: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms\n";


Tensor &intermediate = workspace.intermediate;

start = std::chrono::high_resolution_clock::now();

linear(attention_dense, intermediate_weight, intermediate_bias, intermediate);

end = std::chrono::high_resolution_clock::now();
std::cout << "Intermediate linear: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms\n";


start = std::chrono::high_resolution_clock::now();

gelu(intermediate);

end = std::chrono::high_resolution_clock::now();
std::cout << "GELU: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms\n";


start = std::chrono::high_resolution_clock::now();

linear(intermediate, output_weight, output_bias, output);

end = std::chrono::high_resolution_clock::now();
std::cout << "Output linear: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms\n";


start = std::chrono::high_resolution_clock::now();

residual(output, attention_dense);

end = std::chrono::high_resolution_clock::now();
std::cout << "Residual 2: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms\n";


start = std::chrono::high_resolution_clock::now();

layer_norm(output, output_layernorm_weight, output_layernorm_bias);

end = std::chrono::high_resolution_clock::now();
std::cout << "LayerNorm 2: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms\n";
}