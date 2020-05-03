#include "env.h"

#include <map>
#include <utility>

bool is_end(const Graph& g) { return g.num_nodes == 0; }

Graph step(const Graph& g, int action) {
    std::map<int, int> nodes;
    int num_nodes = 0;
    for (int a : g.adj_list[action]) {
        nodes[a] = num_nodes++;
    }

    Graph ret(num_nodes);
    for (auto& p : g.edge_list) {
        int u = p.first, v = p.second;
        if (nodes.count(u) && nodes.count(v)) {
            ret.add_edge(nodes[u], nodes[v]);
        }
    }
    return ret;
}
