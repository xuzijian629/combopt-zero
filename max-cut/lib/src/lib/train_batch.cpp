#include "train_batch.h"

#include <vector>

TrainBatch::TrainBatch() : n(0) {}

TrainBatch::TrainBatch(const std::vector<Graph>& graphs, const std::vector<std::vector<int>>& adj_blacks,
                       const std::vector<std::vector<int>>& adj_whites, const std::vector<int>& actions,
                       const std::vector<std::vector<float>>& pis, const std::vector<float>& rewards)
    : n(graphs.size()),
      graphs(graphs),
      adj_blacks(adj_blacks),
      adj_whites(adj_whites),
      actions(actions),
      pis(pis),
      rewards(rewards) {}

template <class T>
void concat(std::vector<T>& a, const std::vector<T>& b) {
    a.insert(a.end(), b.begin(), b.end());
}

void TrainBatch::operator+=(const TrainBatch& rhs) {
    n += rhs.n;
    concat(graphs, rhs.graphs);
    concat(adj_blacks, rhs.adj_blacks);
    concat(adj_whites, rhs.adj_whites);
    concat(actions, rhs.actions);
    concat(pis, rhs.pis);
    concat(rewards, rhs.rewards);
}

void TrainBatch::clear() {
    n = 0;
    graphs.clear();
    adj_blacks.clear();
    adj_whites.clear();
    actions.clear();
    pis.clear();
    rewards.clear();
}

TrainBatch TrainData;
