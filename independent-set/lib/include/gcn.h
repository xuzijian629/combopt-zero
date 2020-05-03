#ifndef GCN_H
#define GCN_H
#include <torch/torch.h>

#include "config.h"

struct GraphConvolutionalNetworkImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    GraphConvolutionalNetworkImpl(int input_dim, int output_dim) {
        int hidden_dim = cfg::gcn_hidden_dim;
        int layer_num = cfg::gcn_layer_num;
        for (int i = 0; i < layer_num; i++) {
            int in = i == 0 ? input_dim : hidden_dim;
            int out = i == layer_num - 1 ? output_dim : hidden_dim;
            std::string module_name = "layer" + std::to_string(i);
            layers.push_back(register_module(module_name, torch::nn::Linear(in, out)));
        }
    }
    // normalize adj by D^(-1/2) * A * D^(-1/2), where A = adj + E
    void normalize_adj(torch::Tensor adj) {
        int n = adj.sizes()[0];
        for (int i = 0; i < n; i++) {
            adj[i][i] += 1;
        }
        torch::Tensor deg = adj.sum(0);
        torch::Tensor coef_mat = torch::t(deg) * deg;
        adj /= coef_mat;
    }
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
        int n = x.sizes()[0];
        assert(adj.sizes() == std::vector<int64_t>({n, n}));
        normalize_adj(adj);
        for (int i = 0; i < (int)layers.size(); i++) {
            x = torch::matmul(adj, x);
            x = layers[i]->forward(x);
        }
        return x;
    }
};
TORCH_MODULE(GraphConvolutionalNetwork);
#endif
