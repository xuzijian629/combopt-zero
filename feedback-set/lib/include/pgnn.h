#ifndef PGNN_H
#define PGNN_H
#include <torch/torch.h>

#include "config.h"

struct MatrixMLPImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> linears;
    MatrixMLPImpl(int input_dim, int output_dim, int hidden_dim, int linear_num) {
        for (int i = 0; i < linear_num; i++) {
            int in = i == 0 ? input_dim : hidden_dim;
            int out = i == linear_num - 1 ? output_dim : hidden_dim;
            std::string module_name = "linear" + std::to_string(i);
            linears.push_back(register_module(module_name, torch::nn::Linear(in, out)));
        }
    }
    torch::Tensor forward(torch::Tensor x) {
        int n = x.sizes()[0];
        assert(x.sizes().size() == 3);
        assert(x.sizes().size() == 3 && x.sizes()[1] == n);
        x = x.reshape({n * n, -1});
        for (int i = 0; i < (int)linears.size(); i++) {
            x = linears[i]->forward(x);
            if (i != (int)linears.size() - 1) x = torch::relu(x);
        }
        return x.reshape({n, n, -1});
    }
};
TORCH_MODULE(MatrixMLP);

struct MatrixProductLayerImpl : torch::nn::Module {
    MatrixMLP m1, m2, m4;  // m3 is just an identity function
    MatrixProductLayerImpl(int input_dim, int output_dim)
        : m1(register_module("m1",
                             MatrixMLP(input_dim, output_dim, cfg::pgnn_mlp_hidden_dim, cfg::pgnn_mlp_layer_num))),
          m2(register_module("m2",
                             MatrixMLP(input_dim, output_dim, cfg::pgnn_mlp_hidden_dim, cfg::pgnn_mlp_layer_num))),
          m4(register_module(
              "m4", MatrixMLP(input_dim + output_dim, output_dim, cfg::pgnn_mlp_hidden_dim, cfg::pgnn_mlp_layer_num))) {
    }

    // [n, n, a(=input_dim)] -> [n, n, b(=output_dim)]
    torch::Tensor forward(torch::Tensor adj) {
        // adj: [n, n, a]
        torch::Tensor x1 = m1(adj).permute({2, 0, 1});               // [a, n, n]
        torch::Tensor x2 = m2(adj).permute({2, 0, 1});               // [a, n, n]
        torch::Tensor tmp1 = torch::bmm(x1, x2).permute({1, 2, 0});  // [n, n, a]
        torch::Tensor tmp2 = torch::cat({tmp1, adj}, 2);             // [a + b, n, n]
        return m4(tmp2);
    }
};
TORCH_MODULE(MatrixProductLayer);

struct Equivariant2to1Impl : torch::nn::Module {
    torch::nn::Linear linear;
    Equivariant2to1Impl(int input_dim, int output_dim)
        : linear(register_module("linear", torch::nn::Linear(5 * input_dim, output_dim))) {}
    // [n, n, d] -> [n, 5, d]
    torch::Tensor concat_bases(torch::Tensor adj, bool normalize = true) {
        int n = adj.sizes()[0], d = adj.sizes()[2];
        torch::Tensor diag = adj.diagonal(0, 0, 1).reshape({n, 1, d});  // [n, 1, d]

        // op1-5 : [n, 1, d]
        // op1 - (123) - extract diag
        torch::Tensor op1 = diag;
        // op2 - (123) + (12)(3) - tile sum of diag part
        torch::Tensor op2 = diag.sum(0, true).repeat({n, 1, 1});
        // op3 - (123) + (13)(2) - place sum of row i in element i
        torch::Tensor op3 = adj.sum(1, true);
        // op4 - (123) + (23)(1) - place sum of col i in element i
        torch::Tensor op4 = adj.sum(0, true).view({n, 1, d});
        // op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        torch::Tensor op5 = op3.sum(0, true).repeat({n, 1, 1});

        // normalize by the number of entries
        if (normalize) {
            op2 /= n;
            op3 /= n;
            op4 /= n;
            op5 /= n * n;
        }
        return torch::cat({op1, op2, op3, op4, op5}, 1);
    }
    // [n, n, d] -> [n, output_dim]
    torch::Tensor forward(torch::Tensor adj) {
        int n = adj.sizes()[0], d = adj.sizes()[2];
        assert(adj.sizes().size() == 3 && adj.sizes()[1] == n);
        torch::Tensor bases = concat_bases(adj);  // [n, 5, d]
        return linear(bases.view({n, 5 * d}));
    }
};
TORCH_MODULE(Equivariant2to1);

struct PGNNImpl : torch::nn::Module {
    std::vector<MatrixProductLayer> mps;
    std::vector<Equivariant2to1> equivs;
    int output_dim;
    PGNNImpl(int input_dim, int output_dim) : output_dim(output_dim) {
        for (int i = 0; i < cfg::pgnn_layer_num; i++) {
            int in = i == 0 ? input_dim : cfg::pgnn_hidden_dim;
            int out = cfg::pgnn_hidden_dim;
            std::string mp_name = "MP" + std::to_string(i);
            mps.push_back(register_module(mp_name, MatrixProductLayer(in, out)));
            std::string equiv_name = "equiv" + std::to_string(i);
            equivs.push_back(register_module(equiv_name, Equivariant2to1(out, output_dim)));
        }
    }

    // NOTE: this network does not use first argument
    // (_, [n, n, input_dim]) -> [n, output_dim]
    torch::Tensor forward(torch::Tensor _, torch::Tensor adj) {
        int n = adj.sizes()[0];
        assert(adj.sizes() == std::vector<int64_t>({n, n}));
        torch::Tensor scores = torch::zeros({n, output_dim}).to(cfg::device);
        torch::Tensor cur = adj.reshape({n, n, 1});
        for (int i = 0; i < (int)mps.size(); i++) {
            cur = mps[i](cur);
            scores = scores + equivs[i](cur);
        }
        return scores;
    }
};
TORCH_MODULE(PGNN);

#endif