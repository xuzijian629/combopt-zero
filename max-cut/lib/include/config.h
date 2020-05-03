#ifndef cfg_H
#define cfg_H
#include <torch/torch.h>

#include <string>

struct cfg {
    static torch::Device device;
    static int max_n, min_n;
    static int batch_size;
    static float batch_times;
    static int initial_phase_len;
    static float alpha;
    static float beta;
    static int use_sigmoid;
    static float dirichlet_alpha;
    static float dirichlet_eps;
    static int num_play;
    static int max_n_log;
    static int rollout_coef;
    static int min_rollout;
    static int max_rollout;
    static float learning_rate;
    static float weight_decay;
    static int s2v_embed_dim;
    static int s2v_reg_hidden;
    static int s2v_iter;
    static int gin_hidden_dim;
    static int gin_layer_num;
    static int gin_mlp_hidden_dim;
    static int gin_mlp_layer_num;
    static int pgnn_layer_num;
    static int pgnn_hidden_dim;
    static int pgnn_mlp_hidden_dim;
    static int pgnn_mlp_layer_num;
    static int gcn_hidden_dim;
    static int gcn_layer_num;
    static int gat_hidden_num;
    static int gat_head_num;
    static int gat_layer_num;
    static int gat_leakyrelu_alpha;
    static constexpr float eps = 1e-3;
    static std::string save_dir;
    static std::string gnn_type;
    static int add_noise_test;

    static void LoadParams(const int argc, const char** argv);
};

#endif
