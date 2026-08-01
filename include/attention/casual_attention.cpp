
#include "attention/casual_attention.hpp"
#include "utils/conversion.hpp"

float dot_product(
    const float* q,
    const float* k,
    int head_dim)
{
    float sum = 0.0f;

    for (int i = 0; i < head_dim; i++)
        sum += q[i] * k[i];

    return sum;
}

int Attention::load(const std::string file_path)
{
    tensor_loader.load(file_path);
    input_layernorm = tensor_loader.get_tensor("model.layers.0.input_layernorm.weight");
    mlp_proj_down = tensor_loader.get_tensor("model.layers.0.mlp.down_proj.weight");
    mlp_gate_proj = tensor_loader.get_tensor("model.layers.0.mlp.gate_proj.weight");
    mlp_up_proj = tensor_loader.get_tensor("model.layers.0.mlp.up_proj.weight");
    post_attention_layernorm = tensor_loader.get_tensor("model.layers.0.post_attention_layernorm.weight");
    k_norm = tensor_loader.get_tensor("model.layers.0.self_attn.k_norm.weight");
    k_proj = tensor_loader.get_tensor("model.layers.0.self_attn.k_proj.weight");
    o_proj = tensor_loader.get_tensor("model.layers.0.self_attn.o_proj.weight");
    q_norm = tensor_loader.get_tensor("model.layers.0.self_attn.q_norm.weight");
    q_proj = tensor_loader.get_tensor("model.layers.0.self_attn.q_proj.weight");
    v_proj = tensor_loader.get_tensor("model.layers.0.self_attn.v_proj.weight");
    return 0;
}

void Attention::forward(
    const std::vector<std::vector<float>> &tokens,
    std::vector<std::vector<float>> &output)
{

    int i = 0;
    std::vector<std::vector<float>> Q;
    std::vector<std::vector<float>> K;
    std::vector<std::vector<float>> V;
    for (const std::vector<float> token : tokens){
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
    
        linear(q_proj, token, q);
        linear(k_proj, token, k);
        linear(v_proj, token, v);
    
        // RMS Norm
        rms_norm(q, q_norm, 80);
        rms_norm(k, k_norm, 80);

        // RoPE
        apply_rope(q, 80, i);
        apply_rope(k, 80, i);

        Q.push_back(std::move(q));
        K.push_back(std::move(k));
        V.push_back(std::move(v));
        i++;
    }
    output.push_back(q);
}

void Attention::linear(
    const Tensor &weight,
    const std::vector<float> &input,
    std::vector<float> &output)
{
    int out_dim = weight.shape[0];
    int in_dim = weight.shape[1];

    if (input.size() != static_cast<size_t>(in_dim))
    {
        throw std::runtime_error(
            "Linear input dimension mismatch. Expected " + std::to_string(in_dim) + ", got " + std::to_string(input.size()));
    }

    output.resize(out_dim);

    uint16_t *w = reinterpret_cast<uint16_t *>(weight.data);

    for (int i = 0; i < out_dim; i++)
    {
        float sum = 0.0f;

        for (int j = 0; j < in_dim; j++)
        {
            float weight_value = bf16_to_float(w[i * in_dim + j]);
            sum += weight_value * input[j];
        }

        output[i] = sum;
    }
}

void Attention::rms_norm(
    std::vector<float> &x,
    const Tensor &weight,
    int head_dim,
    float eps = 1e-6f)
{
    const uint16_t *w = reinterpret_cast<const uint16_t *>(weight.data);

    int num_heads = x.size() / head_dim;

    for (int h = 0; h < num_heads; h++)
    {
        size_t offset = h * head_dim;
        float mean_square = 0.0f;

        for (int i = 0; i < head_dim; i++)
        {
            float value = x[offset + i];
            mean_square += value * value;
        }

        mean_square /= head_dim;

        float inv_rms = 1.0f / std::sqrt(mean_square + eps);

        // Normalize and apply learned weight
        for (int i = 0; i < head_dim; i++)
        {
            x[offset + i] *= inv_rms;
            x[offset + i] *= bf16_to_float(w[i]);
        }
    }
}

void Attention::apply_rope(
    std::vector<float> &x,
    int head_dim,
    int position,
    float theta = 1000000.0f)
{
    int num_heads = x.size() / head_dim;
    int half = head_dim / 2;

    for (int h = 0; h < num_heads; h++)
    {
        float *head = x.data() + h * head_dim;

        for (int i = 0; i < half; i++)
        {
            float freq = std::pow(theta, -2.0f * i / head_dim);
            float angle = position * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float x0 = head[i];
            float x1 = head[i + half];

            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}