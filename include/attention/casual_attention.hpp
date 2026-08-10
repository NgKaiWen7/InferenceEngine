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

    void rms_norm(
        const Tensor &weight,
        std::vector<float> &x,
        int head_dim);

    void apply_rope(
        std::vector<float> &x,
        int head_dim,
        int position);

    void attention();

    void softmax(const std::vector<float>& input, std::vector<float>& output);

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