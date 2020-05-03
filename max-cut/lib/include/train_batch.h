#ifndef TRAIN_BATCH_H
#define TRAIN_BATCH_H

#include <vector>

#include "graph.h"

struct TrainBatch {
    int n;
    std::vector<Graph> graphs;
    std::vector<std::vector<int>> adj_blacks, adj_whites;
    std::vector<int> actions;
    std::vector<std::vector<float>> pis;
    std::vector<float> rewards;
    TrainBatch();
    TrainBatch(const std::vector<Graph>& graphs, const std::vector<std::vector<int>>& adj_blacks,
               const std::vector<std::vector<int>>& adj_whites, const std::vector<int>& actions,
               const std::vector<std::vector<float>>& pis, const std::vector<float>& rewards);
    void operator+=(const TrainBatch& rhs);
    void clear();
};

extern TrainBatch TrainData;

#endif
