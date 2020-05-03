#include "mock_policy.h"

#include <torch/torch.h>

#include <iostream>

#include "policy.h"

MockPolicy::MockPolicy() : Policy() {}

std::pair<torch::Tensor, torch::Tensor> MockPolicy::infer_one(const Graph& graph, const std::vector<int>& adj_black,
                                                              const std::vector<int>& adj_white, bool no_grad = true) {
    int num_nodes = graph.num_nodes;
    torch::Tensor policy = torch::zeros({num_nodes});
    torch::Tensor value = torch::zeros({num_nodes});
    return {policy, value};
}

float MockPolicy::train(const std::vector<int>& idxes) { return 1.0; }

void MockPolicy::save(const std::string& filename) {}
void MockPolicy::load(const std::string& filename) {}
