#include "env.h"

#include <map>
#include <utility>

bool is_end(const Graph& g) { return g.num_nodes == 0; }

Graph step(const Graph& g, int action) {
    std::map<int, int> avail_nodes;
    for (int i = 0; i < g.num_nodes; i++) avail_nodes[i];
    for (int a : g.adj_list[action]) avail_nodes.erase(a);
    avail_nodes.erase(action);
    int num_nodes = 0;
    for (auto& p : avail_nodes) p.second = num_nodes++;
    Graph ret(num_nodes);
    for (auto& p : g.edge_list) {
        int u = p.first, v = p.second;
        if (avail_nodes.count(u) && avail_nodes.count(v)) {
            ret.add_edge(avail_nodes[u], avail_nodes[v]);
        }
    }

    return ret;
}
