#ifndef MCTS_H
#define MCTS_H

#include <string>
#include <unordered_map>
#include <vector>

#include "graph.h"
#include "hash.h"
#include "util.h"

class Node {
    void init();
    NodeInfo get_mean_std();
    GNNInfo get_gnn_estimate();

public:
    hash_t adj_hash;
    std::shared_ptr<Node> parent;
    int last_action;
    Graph graph;
    int num_nodes;
    float reward_mean, reward_std;
    std::vector<std::shared_ptr<Node>> children;
    std::vector<int> visit_cnt;
    int visit_cnt_sum;
    std::vector<float> policy, value;

    Node();
    Node(const Graph& graph);
    Node(std::shared_ptr<Node> parent, int last_action);
    int best_child(bool add_noise);
    float state_value();
    std::vector<float> pi(float tau);
};

int random_play(const Graph& g);

float train();
int test();
int test_by_mcts();

void generate_train_data(const std::string& filename);

#endif
