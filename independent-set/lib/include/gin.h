#ifndef GIN_H
#define GIN_H
#include <torch/torch.h>

#include "config.h"

// Because of libtorch macro `TORCH_MODULE`, implementation is included in this header file.

struct MultiLayerPerceptronImpl : torch::nn::Module {
    // TODO: Module::forwardがなくてModuleListの使い方がわからないのでstd::vectorで代用
    std::vector<torch::nn::Linear> linears;
    MultiLayerPerceptronImpl(int input_dim, int output_dim, int hidden_dim, int linear_num) {
        for (int i = 0; i < linear_num; i++) {
            int in = i == 0 ? input_dim : hidden_dim;
            int out = i == linear_num - 1 ? output_dim : hidden_dim;
            std::string module_name = "linear" + std::to_string(i);
            linears.push_back(register_module(module_name, torch::nn::Linear(in, out)));
        }
    }
    torch::Tensor forward(torch::Tensor x) {
        for (int i = 0; i < (int)linears.size(); i++) {
            x = linears[i]->forward(x);
            if (i != (int)linears.size() - 1) x = torch::relu(x);
        }
        return x;
    }
};
TORCH_MODULE(MultiLayerPerceptron);

struct GraphIsomorphismNetworkImpl : torch::nn::Module {
    std::vector<MultiLayerPerceptron> layers;
    MultiLayerPerceptron last_mlp = nullptr;
    GraphIsomorphismNetworkImpl(int input_dim, int output_dim) {
        int hidden_dim = cfg::gin_hidden_dim;
        int layer_num = cfg::gin_layer_num;
        int mlp_hidden_dim = cfg::gin_mlp_hidden_dim;
        int mlp_layer_num = cfg::gin_mlp_layer_num;
        for (int i = 0; i < layer_num; i++) {
            int in = i == 0 ? input_dim : hidden_dim;
            int out = hidden_dim;
            std::string module_name = "layer" + std::to_string(i);
            layers.push_back(
                register_module(module_name, MultiLayerPerceptron(in, out, mlp_hidden_dim, mlp_layer_num)));
        }
        int concatenated_dim = input_dim + hidden_dim * layer_num;
        last_mlp = register_module("last_mlp_layer",
                                   MultiLayerPerceptron(concatenated_dim, output_dim, mlp_hidden_dim, mlp_layer_num));
    }
    torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
        int n = x.sizes()[0];
        assert(adj.sizes() == std::vector<int64_t>({n, n}));
        torch::Tensor acc = x.clone().to(cfg::device);
        for (int i = 0; i < (int)layers.size(); i++) {
            // x += torch::matmul(adj, x);  // backwardが動かない！
            x = x + torch::matmul(adj, x);  // note: self-loop
            x = layers[i]->forward(x);
            acc = torch::cat({acc, x}, 1);
        }
        return last_mlp->forward(acc);
    }
};
TORCH_MODULE(GraphIsomorphismNetwork);
#endif
