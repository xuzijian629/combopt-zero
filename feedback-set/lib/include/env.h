#ifndef ENV_H
#define ENV_H

#include "graph.h"

struct UnionFind {
    int n, cnt;
    std::vector<int> par, rank;
    UnionFind(int n) : n(n), cnt(n), par(n), rank(n) {
        for (int i = 0; i < n; i++) par[i] = i;
    }
    int find(int x) {
        if (x == par[x]) return x;
        return par[x] = find(par[x]);
    }
    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (rank[x] < rank[y]) {
            par[x] = y;
        } else {
            par[y] = x;
            if (rank[x] == rank[y]) rank[x]++;
        }
        cnt--;
        return true;
    }
};

bool is_end(const Graph& g);
Graph step(const Graph& g, int action);

#endif
