#ifndef ENV_H
#define ENV_H

#include <vector>

#include "graph.h"

bool is_end(const Graph& g);
Graph step(const Graph& g, int action, std::vector<int>& adj_black, std::vector<int>& adj_white, int& reward);

#endif
