#include "config.h"

#include <torch/torch.h>

#include <cstring>
#include <iostream>

torch::Device cfg::device = torch::kCPU;
int cfg::min_n;
int cfg::max_n;
int cfg::batch_size;
float cfg::batch_times;
int cfg::initial_phase_len;
float cfg::alpha;
float cfg::beta;
int cfg::use_sigmoid;
float cfg::dirichlet_alpha;
float cfg::dirichlet_eps;
int cfg::num_play;
int cfg::max_n_log;
int cfg::rollout_coef;
int cfg::min_rollout;
int cfg::max_rollout;
float cfg::learning_rate;
float cfg::weight_decay;

// s2v
int cfg::s2v_embed_dim;
int cfg::s2v_reg_hidden;
int cfg::s2v_iter;

// gin
int cfg::gin_hidden_dim;
int cfg::gin_layer_num;
int cfg::gin_mlp_hidden_dim;
int cfg::gin_mlp_layer_num;

// pgnn
int cfg::pgnn_hidden_dim;
int cfg::pgnn_layer_num;
int cfg::pgnn_mlp_hidden_dim;
int cfg::pgnn_mlp_layer_num;

// gcn
int cfg::gcn_hidden_dim;
int cfg::gcn_layer_num;

// gat
int cfg::gat_hidden_num;
int cfg::gat_head_num;
int cfg::gat_layer_num;
int cfg::gat_leakyrelu_alpha;

std::string cfg::save_dir;
std::string cfg::gnn_type;
int cfg::add_noise_test;

void cfg::LoadParams(const int argc, const char** argv) {
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-gpu_id") == 0 && torch::cuda::is_available())
            device = torch::Device(torch::kCUDA, atoi(argv[i + 1]));
        if (strcmp(argv[i], "-max_n") == 0) max_n = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-min_n") == 0) min_n = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-batch_size") == 0) batch_size = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-batch_times") == 0) batch_times = atof(argv[i + 1]);
        if (strcmp(argv[i], "-initial_phase_len") == 0) initial_phase_len = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-alpha") == 0) alpha = atof(argv[i + 1]);
        if (strcmp(argv[i], "-beta") == 0) beta = atof(argv[i + 1]);
        if (strcmp(argv[i], "-use_sigmoid") == 0) use_sigmoid = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-dirichlet_alpha") == 0) dirichlet_alpha = atof(argv[i + 1]);
        if (strcmp(argv[i], "-dirichlet_eps") == 0) dirichlet_eps = atof(argv[i + 1]);
        if (strcmp(argv[i], "-num_play") == 0) num_play = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-max_n_log") == 0) max_n_log = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-rollout_coef") == 0) rollout_coef = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-min_rollout") == 0) min_rollout = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-max_rollout") == 0) max_rollout = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-learning_rate") == 0) learning_rate = atof(argv[i + 1]);
        if (strcmp(argv[i], "-weight_decay") == 0) weight_decay = atof(argv[i + 1]);
        if (strcmp(argv[i], "-s2v_embed_dim") == 0) s2v_embed_dim = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-s2v_reg_hidden") == 0) s2v_reg_hidden = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-s2v_iter") == 0) s2v_iter = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gin_hidden_dim") == 0) gin_hidden_dim = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gin_layer_num") == 0) gin_layer_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gin_mlp_hidden_dim") == 0) gin_mlp_hidden_dim = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gin_mlp_layer_num") == 0) gin_mlp_layer_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-pgnn_hidden_dim") == 0) pgnn_hidden_dim = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-pgnn_layer_num") == 0) pgnn_layer_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-pgnn_mlp_hidden_dim") == 0) pgnn_mlp_hidden_dim = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-pgnn_mlp_layer_num") == 0) pgnn_mlp_layer_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gcn_hidden_dim") == 0) gcn_hidden_dim = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gcn_layer_num") == 0) gcn_layer_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gat_hidden_num") == 0) gat_hidden_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gat_head_num") == 0) gat_head_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gat_layer_num") == 0) gat_layer_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-gat_leakyrelu_alpha") == 0) gat_leakyrelu_alpha = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-save_dir") == 0) {
            save_dir = std::string(argv[i + 1]);
            if (save_dir.back() != '/') save_dir.push_back('/');
        }
        if (strcmp(argv[i], "-gnn_type") == 0) gnn_type = std::string(argv[i + 1]);
        if (strcmp(argv[i], "-add_noise_test") == 0) add_noise_test = atoi(argv[i + 1]);
    }
}
