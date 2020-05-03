#include "hash.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>

#include "config.h"
#include "graph.h"
using Long = __uint128_t;

// 2 is a primitive root of MOD
constexpr hash_t MOD = 1000000000000000009;
std::vector<hash_t> pow2, pow13333;
std::unordered_map<int, hash_t> memo;

void init_hash() {
    hash_t acc = 0;
    hash_t b = 1;
    std::map<int, int> save;
    int max_n = (1 << cfg::max_n_log) - 1;
    pow2.resize(max_n + 1);
    pow13333.resize(max_n + 1);
    pow2[0] = pow13333[0] = 1;
    for (int i = 1; i <= max_n; i++) {
        pow2[i] = pow2[i - 1] * 2 % MOD;
        pow13333[i] = (Long)pow13333[i - 1] * 13333 % MOD;
    }
    for (int i = 0; i <= max_n; i++) save[i * (i - 1) / 2] = i;
    for (int i = 0; i <= max_n * (max_n - 1) / 2; i++) {
        if (save.count(i)) memo[save[i]] = acc;
        acc += b;
        if (acc >= MOD) acc -= MOD;
        b <<= 1;
        if (b >= MOD) b -= MOD;
    }
}

hash_t get_adj_hash_labeled(const Graph& graph, const std::vector<int>& adj_black, const std::vector<int>& adj_white) {
    hash_t ret = memo[graph.num_nodes];
    for (auto& p : graph.edge_list) {
        int u = p.first, v = p.second;
        int k = (2 * graph.num_nodes - 1 - u) * u / 2 + v - 1 - u;
        ret += pow2[k];
        if (ret >= MOD) ret -= MOD;
    }

    for (int i = 0; i < graph.num_nodes; i++) {
        ret += ((adj_black[i] << cfg::max_n_log) + adj_white[i]) * pow13333[i] % MOD;
        if (ret >= MOD) ret -= MOD;
    }

    assert(graph.num_nodes < 1 << cfg::max_n_log);
    return (ret << cfg::max_n_log) | graph.num_nodes;
}

hash_t get_adj_hash_dense_labeled(const Graph& graph, const std::vector<int>& adj_black,
                                  const std::vector<int>& adj_white) {
    hash_t ret = 0;
    hash_t b = 1;
    int n = graph.num_nodes;
    std::vector<std::vector<int>> adj(n, std::vector<int>(n));
    for (auto& p : graph.edge_list) {
        int u = p.first, v = p.second;
        adj[u][v]++;
    }
    for (int i = 0; i < graph.num_nodes; i++) {
        for (int j = i + 1; j < graph.num_nodes; j++) {
            ret += (adj[i][j] + 1) * b;
            while (ret >= MOD) ret -= MOD;
            b <<= 1;
            if (b >= MOD) b -= MOD;
        }
    }

    for (int i = 0; i < graph.num_nodes; i++) {
        ret += ((adj_black[i] << cfg::max_n_log) + adj_white[i]) * pow13333[i] % MOD;
        if (ret >= MOD) ret -= MOD;
    }

    return (ret << cfg::max_n_log) | graph.num_nodes;
}
