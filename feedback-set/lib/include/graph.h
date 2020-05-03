#ifndef GRAPH_H
#define GRAPH_H

#include <torch/torch.h>

#include <map>
#include <vector>

class Graph {
public:
    Graph();
    Graph(const int num_nodes);
    int num_nodes;
    int num_edges;
    std::vector<std::vector<int>> adj_list;
    std::vector<std::pair<int, int>> edge_list;

    void add_edge(int a, int b);
    torch::Tensor to_adj_tensor() const;
};

extern Graph CurrentGraph, CurrentTestGraph;

#endif
