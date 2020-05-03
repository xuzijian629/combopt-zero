#ifndef HASH_H
#define HASH_H

#include <set>
#include <unordered_map>
#include <vector>

#include "graph.h"

typedef unsigned long long hash_t;

void init_hash();
hash_t get_adj_hash_labeled(const Graph& graph, const std::vector<int>& adj_black, const std::vector<int>& adj_white);
hash_t get_adj_hash_dense_labeled(const Graph& graph, const std::vector<int>& adj_black,
                                  const std::vector<int>& adj_white);

typedef std::pair<float, float> NodeInfo;
typedef std::pair<std::vector<float>, std::vector<float>> GNNInfo;

#endif
