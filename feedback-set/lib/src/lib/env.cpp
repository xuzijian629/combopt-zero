#include "env.h"

#include <map>
#include <utility>

bool is_end(const Graph& g) {
    int n = g.num_nodes;
    UnionFind uf(n);
    for (auto& p : g.edge_list) {
        if (!uf.unite(p.first, p.second)) return false;
    }
    return true;
}

Graph step(const Graph& g, int action) {
    std::vector<std::pair<int, int>> edges;
    std::map<int, int> nodes;
    for (auto& p : g.edge_list) {
        int u = p.first, v = p.second;
        if (u == action || v == action) continue;
        edges.emplace_back(u, v);
        nodes[u];
        nodes[v];
    }

    int num_nodes = 0;
    for (auto& p : nodes) {
        p.second = num_nodes++;
    }

    Graph ret(num_nodes);
    for (auto& e : edges) {
        ret.add_edge(nodes[e.first], nodes[e.second]);
    }
    return ret;
}
