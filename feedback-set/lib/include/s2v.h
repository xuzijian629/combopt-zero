#ifndef S2V_H
#define S2V_H

#include <torch/torch.h>

#include <cassert>

#include "config.h"

constexpr int aux_dim = 0;

struct S2VImpl : torch::nn::Module {
    S2VImpl(int input_dim, int output_dim)
        : w_n2l(input_dim, cfg::s2v_embed_dim),
          p_node_conv(cfg::s2v_embed_dim, cfg::s2v_embed_dim),
          h1_weight(cfg::s2v_embed_dim, cfg::s2v_reg_hidden),
          h2_weight(cfg::s2v_reg_hidden + aux_dim, output_dim) {
        init_weight(w_n2l);
        init_weight(p_node_conv);
        init_weight(h1_weight);
        init_weight(h2_weight);
        register_module("w_n2l", w_n2l);
        register_module("p_node_conv", p_node_conv);
        register_module("h1_weight", h1_weight);
        register_module("h2_weight", h2_weight);
    }

    void init_weight(torch::nn::Linear &param) {
        torch::nn::init::uniform_(param->weight, -0.08, 0.08);
        torch::nn::init::uniform_(param->bias, -0.08, 0.08);
    }

    torch::nn::Linear w_n2l, p_node_conv, h1_weight, h2_weight;

    torch::Tensor forward(torch::Tensor node_input, torch::Tensor aux_input, torch::Tensor adj) {
        // node_input * w_n2l
        auto input_message = w_n2l(node_input);
        auto input_potential_layer = torch::relu(input_message);
        auto cur_message_layer = input_potential_layer;
        for (int i = 0; i < cfg::s2v_iter; i++) {
            auto n2npool = torch::mm(adj, cur_message_layer);
            // n2npool * p_node_conv
            auto node_linear = p_node_conv(n2npool);
            auto merged_linear = node_linear + input_message;
            cur_message_layer = torch::relu(merged_linear);
        }
        auto hidden = h1_weight(cur_message_layer);
        auto last_output = torch::relu(hidden);
        auto rep_aux = aux_input.repeat({node_input.sizes()[0], aux_dim});
        last_output = torch::cat({last_output, rep_aux}, 1);
        return h2_weight(last_output);
    }

    torch::Tensor forward(torch::Tensor node_input, torch::Tensor adj) {
        return forward(node_input, torch::zeros(aux_dim).to(cfg::device), adj);
    }
};
TORCH_MODULE(S2V);

#endif
