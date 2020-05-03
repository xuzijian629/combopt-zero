#include "graph.h"

#include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <random>

#include "config.h"

Graph::Graph() {}

Graph::Graph(const int num_nodes) : num_nodes(num_nodes), num_edges(0), adj_list(num_nodes) {}

void Graph::add_edge(int a, int b) {
    assert(0 <= a && a < num_nodes);
    assert(0 <= b && b < num_nodes);
    if (a > b) std::swap(a, b);
    adj_list[a].push_back(b);
    adj_list[b].push_back(a);
    edge_list.emplace_back(a, b);
    num_edges++;
}

torch::Tensor Graph::to_adj_tensor() const {
    torch::Tensor adj = torch::zeros({num_nodes, num_nodes});
    for (const auto& edge : edge_list) {
        adj[edge.first][edge.second] = adj[edge.second][edge.first] = 1;
    }
    return adj.to(cfg::device);
}

Graph CurrentGraph, CurrentTestGraph;
