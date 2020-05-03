#ifndef GNN_POLICY_H
#define GNN_POLICY_H

#include <torch/torch.h>

#include "config.h"
#include "policy.h"
#include "train_batch.h"
#include "util.h"

torch::Tensor make_input_feat(const Graph& graph, const std::vector<int>& adj_black,
                              const std::vector<int>& adj_white) {
    int n = graph.num_nodes;
    std::vector<float> feat(n * 3);
    for (int i = 0; i < n; i++) {
        feat[i * 3] = 1;
        feat[i * 3 + 1] = adj_black[i];
        feat[i * 3 + 2] = adj_white[i];
    }
    return from_float32_array(&(*(feat.begin())), {n, 3});
}

template <class GNN>
class GNNPolicy : public Policy {
public:
    GNN gnn;
    torch::optim::Adam optimizer;
    // TODO: Adamの初期化もっときれいに書けそう
    GNNPolicy()
        : gnn(3, 4),
          optimizer(
              gnn->parameters(),
              torch::optim::AdamOptions(cfg::learning_rate).weight_decay(cfg::weight_decay)) {  // input_dim, output_dim
        gnn->to(cfg::device);
        optimizer = torch::optim::Adam(gnn->parameters(),
                                       torch::optim::AdamOptions(cfg::learning_rate).weight_decay(cfg::weight_decay));
    }
    std::pair<torch::Tensor, torch::Tensor> infer_one(const Graph& graph, const std::vector<int>& adj_black,
                                                      const std::vector<int>& adj_white, bool no_grad = true) override {
        assert(graph.num_nodes > 0);
        if (no_grad) {
            torch::NoGradGuard guard;
            torch::Tensor x = make_input_feat(graph, adj_black, adj_white);
            torch::Tensor y = gnn->forward(x, graph.to_adj_tensor());
            torch::Tensor policy_pred_black = y.narrow(1, 0, 1).reshape({-1});
            torch::Tensor policy_pred_white = y.narrow(1, 1, 1).reshape({-1});
            torch::Tensor policy_pred = torch::cat({policy_pred_black, policy_pred_white}, 0);
            policy_pred = policy_pred.softmax(0);
            torch::Tensor value_pred_black = y.narrow(1, 2, 1).reshape({-1});
            torch::Tensor value_pred_white = y.narrow(1, 3, 1).reshape({-1});
            torch::Tensor value_pred = torch::cat({value_pred_black, value_pred_white}, 0);
            value_pred = normalize(value_pred);
            return {policy_pred, value_pred};
        } else {
            torch::Tensor x = make_input_feat(graph, adj_black, adj_white);
            torch::Tensor y = gnn->forward(x, graph.to_adj_tensor());
            torch::Tensor policy_pred_black = y.narrow(1, 0, 1).reshape({-1});
            torch::Tensor policy_pred_white = y.narrow(1, 1, 1).reshape({-1});
            torch::Tensor policy_pred = torch::cat({policy_pred_black, policy_pred_white}, 0);
            policy_pred = policy_pred.softmax(0);
            torch::Tensor value_pred_black = y.narrow(1, 2, 1).reshape({-1});
            torch::Tensor value_pred_white = y.narrow(1, 3, 1).reshape({-1});
            torch::Tensor value_pred = torch::cat({value_pred_black, value_pred_white}, 0);
            value_pred = normalize(value_pred);
            return {policy_pred, value_pred};
        }
    }

    float train(const std::vector<int>& idxes) override {
        auto& graphs = TrainData.graphs;
        auto& adj_blacks = TrainData.adj_blacks;
        auto& adj_whites = TrainData.adj_whites;
        auto& actions = TrainData.actions;
        auto& policy_target = TrainData.pis;
        auto& value_target = TrainData.rewards;
        optimizer.zero_grad();
        torch::Tensor loss = torch::tensor(0.0).to(cfg::device);
        for (int idx : idxes) {
            torch::Tensor policy, value;
            std::tie(policy, value) = infer_one(graphs[idx], adj_blacks[idx], adj_whites[idx], false);
            // policy
            torch::Tensor policy_target_tensor = from_float32_vector(policy_target[idx]);
            assert(policy.sizes()[0] == policy_target_tensor.sizes()[0]);
            loss += cross_entropy(policy, policy_target_tensor);
            // value
            loss += (value_target[idx] - value[actions[idx]]).pow(2);
        }
        loss.backward();
        optimizer.step();
        float ret = loss.item().toFloat();
        assert(!std::isnan(ret));
        return ret;
    }
    void save(const std::string& filename) override {
        gnn->to(torch::kCPU);
        torch::save(gnn, filename);
        gnn->to(cfg::device);
    }

    void load(const std::string& filename) override {
        gnn->to(torch::kCPU);
        torch::load(gnn, filename);
        gnn->to(cfg::device);
    }
};

#endif
