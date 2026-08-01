#include "safetensors.hpp"

class Attention
{
public:
    int load(const std::string file_path);
    void forward(const std::vector<std::vector<float>> &hidden,
        std::vector<std::vector<float>> &output);

private:
    void linear(
        const Tensor &weight,
        const std::vector<float> &input,
        std::vector<float> &output);

    void rope(
        std::vector<float> &q,
        std::vector<float> &k);

    void Attention::rms_norm(
        std::vector<float> &x,
        const Tensor &weight,
        int head_dim,
        float eps = 1e-6f);

    void Attention::apply_rope(
        std::vector<float> &x,
        int head_dim,
        int position,
        float theta = 1000000.0f);

    void attention();

    SafeTensorLoader tensor_loader;

    Tensor input_layernorm;
    Tensor mlp_proj_down;
    Tensor mlp_gate_proj;
    Tensor mlp_up_proj;
    Tensor post_attention_layernorm;
    Tensor k_norm;
    Tensor k_proj;
    Tensor o_proj;
    Tensor q_norm;
    Tensor q_proj;
    Tensor v_proj;
};