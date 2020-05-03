#ifndef HASH_H
#define HASH_H

#include <set>
#include <unordered_map>

#include "graph.h"

typedef unsigned long long hash_t;

void init_hash();
hash_t get_adj_hash(const Graph& graph);
hash_t get_adj_hash_dense(const Graph& graph);

typedef std::pair<float, float> NodeInfo;
typedef std::pair<std::vector<float>, std::vector<float>> GNNInfo;

#endif
