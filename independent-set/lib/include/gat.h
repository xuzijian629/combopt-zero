#ifndef GAT_H
#define GAT_H
#include <torch/torch.h>

#include "config.h"

struct GraphAttentionLayerImpl : torch::nn::Module {
    torch::nn::Linear W, a;
    int output_dim;
    GraphAttentionLayerImpl(int input_dim, int output_dim)
        : W(register_module("W", torch::nn::Linear(input_dim, output_dim))),
          a(register_module("a", torch::nn::Linear(2 * output_dim, 1))),
          output_dim(output_dim) {}
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
        int n = x.sizes()[0];
        assert(adj.sizes() == std::vector<int64_t>({n, n}) && x.sizes().size() == 2);
        torch::Tensor h = W(x);

        // calculate attention
        torch::Tensor a_input = torch::cat({h.repeat({1, n}).view({n * n, -1}), h.repeat({n, 1})})
                                    .view({n, -1, 2 * output_dim});                            // [n, n, 2 * output_dim]
        torch::Tensor e = torch::leaky_relu(a(a_input).squeeze(2), cfg::gat_leakyrelu_alpha);  // [n, n]
        torch::Tensor zero_vec = -9e15 * torch::ones({n, n}).to(cfg::device);
        torch::Tensor attention = torch::where(adj > 0, e, zero_vec).softmax(1);

        return torch::matmul(attention, h);
    }
};
TORCH_MODULE(GraphAttentionLayer);

struct GraphAttentionNetworkImpl : torch::nn::Module {
    std::vector<std::vector<GraphAttentionLayer>> multi_heads;
    int output_dim;
    GraphAttentionNetworkImpl(int input_dim, int output_dim) : multi_heads(cfg::gat_layer_num), output_dim(output_dim) {
        for (int i = 0; i < cfg::gat_layer_num; i++) {
            int in = i == 0 ? input_dim : cfg::gat_hidden_num * cfg::gat_head_num;
            int out = i == cfg::gat_layer_num - 1 ? output_dim : cfg::gat_hidden_num;
            for (int j = 0; j < cfg::gat_head_num; j++) {
                std::string module_name = "layer" + std::to_string(i) + ", head" + std::to_string(j);
                multi_heads[i].push_back(register_module(module_name, GraphAttentionLayer(in, out)));
            }
        }
    }
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
        int n = x.sizes()[0];
        assert(adj.sizes() == std::vector<int64_t>({n, n}) && x.sizes().size() == 2);
        x = adj.sum(1, true);  // set degree as a feature
        for (int i = 0; i < (int)multi_heads.size(); i++) {
            if (i < (int)multi_heads.size() - 1) {
                torch::Tensor acc = torch::zeros({n, 0}).to(cfg::device);
                for (int j = 0; j < (int)multi_heads[i].size(); j++) {
                    acc = torch::cat({acc, multi_heads[i][j](x, adj)}, 1);
                }
                x = torch::elu(acc);
            } else {
                torch::Tensor acc = torch::zeros({n, output_dim}).to(cfg::device);
                for (int j = 0; j < (int)multi_heads[i].size(); j++) {
                    acc = acc + multi_heads[i][j](x, adj);
                }
                x = acc / (float)multi_heads[i].size();
            }
        }
        return x;
    }
};
TORCH_MODULE(GraphAttentionNetwork);
#endif
