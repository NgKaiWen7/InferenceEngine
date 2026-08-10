
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
        rms_norm(q_norm, q, 80);
        rms_norm(k_norm, k,  80);

        // RoPE
        apply_rope(q, 80, i);
        apply_rope(k, 80, i);

        Q.push_back(std::move(q));
        K.push_back(std::move(k));
        V.push_back(std::move(v));
        i++;
    }

    for (int i = 0; i < Q.size(); i++)
    {
        std::vector<float> scores;

        for (int j = 0; j < K.size(); j++)
        {
            float score = 0.0f;

            for (int d = 0; d < Q[i].size(); d++)
                score += Q[i][d] * K[j][d];

            score /= std::sqrt(Q[i].size());
            scores.push_back(score);
        }
        
        std::vector<float> softmax_scores;
        softmax(scores, softmax_scores);

        std::vector<float> attention_output(V[i].size(), 0.0f);

        for (int j = 0; j < V.size(); j++)
            for (int d = 0; d < V[j].size(); d++)
                attention_output[d] += softmax_scores[j] * V[j][d];

        output.push_back(std::move(attention_output));
    }
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
    const Tensor &weight,
    std::vector<float> &x,
    int head_dim)
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

        float inv_rms = 1.0f / std::sqrt(mean_square + 1e-6f);

        // Apply RMS normalization and the learned weight.
        for (int i = 0; i < head_dim; i++)
        {
            float weight = bf16_to_float(w[i]);
            x[offset + i] = x[offset + i] * inv_rms * weight;
        }
    }
}

void Attention::apply_rope(
    std::vector<float> &x,
    int head_dim,
    int position)
{
    int dim = x.size();

    for (int i = 0; i < dim; i += 2)
    {
        float exponent = static_cast<float>(i) / dim;
        float frequency = 1.0f / std::pow(10000.0f, exponent);
        float angle = position * frequency;

        float cos_theta = std::cos(angle);
        float sin_theta = std::sin(angle);

        float x0 = x[i];
        float x1 = x[i + 1];

        x[i] = x0 * cos_theta - x1 * sin_theta;
        x[i + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

void Attention:: softmax(const std::vector<float>& input, std::vector<float>& output) {
    if (input.empty()) {
        return;
    }
    // 1. Find the maximum value in the input vector to prevent overflow
    float max_val = *std::max_element(input.begin(), input.end());

    output.resize(input.size());
    float sum = 0.0;

    // 2. Compute exponentials shifted by the maximum value
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    // 3. Normalize the values so they sum up to 1
    // Multiplying by the reciprocal is faster than dividing in each iteration
    float inv_sum = 1.0 / sum;
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] *= inv_sum;
    }
    return;
}