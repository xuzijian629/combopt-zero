#include "env.h"

#include <map>
#include <utility>

bool is_end(const Graph& g) { return g.num_nodes == 0; }

Graph step(const Graph& g, int action, std::vector<int>& adj_black, std::vector<int>& adj_white, int& reward) {
    reward = 0;
    int n = g.num_nodes;
    assert(n);
    assert((int)adj_black.size() == n);
    assert((int)adj_white.size() == n);
    int actual_node;
    if (action < n) {
        actual_node = action;
        reward += adj_white[actual_node];
        for (int a : g.adj_list[actual_node]) {
            adj_black[a]++;
        }
    } else {
        actual_node = action - n;
        reward += adj_black[actual_node];
        for (int a : g.adj_list[actual_node]) {
            adj_white[a]++;
        }
    }
    adj_black.erase(adj_black.begin() + actual_node);
    adj_white.erase(adj_white.begin() + actual_node);
    Graph ret(n - 1);
    for (auto& p : g.edge_list) {
        int u = p.first, v = p.second;
        if (u == actual_node || v == actual_node) continue;
        if (u > actual_node) u--;
        if (v > actual_node) v--;
        ret.add_edge(u, v);
    }

    return ret;
}
