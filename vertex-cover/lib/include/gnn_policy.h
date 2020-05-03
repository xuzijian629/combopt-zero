#ifndef GNN_POLICY_H
#define GNN_POLICY_H

#include <torch/torch.h>

#include "config.h"
#include "policy.h"
#include "train_batch.h"
#include "util.h"

template <class GNN>
class GNNPolicy : public Policy {
public:
    GNN gnn;
    torch::optim::Adam optimizer;
    // TODO: Adamの初期化もっときれいに書けそう
    GNNPolicy()
        : gnn(1, 2),
          optimizer(
              gnn->parameters(),
              torch::optim::AdamOptions(cfg::learning_rate).weight_decay(cfg::weight_decay)) {  // input_dim, output_dim
        gnn->to(cfg::device);
        optimizer = torch::optim::Adam(gnn->parameters(),
                                       torch::optim::AdamOptions(cfg::learning_rate).weight_decay(cfg::weight_decay));
    }
    std::pair<torch::Tensor, torch::Tensor> infer_one(const Graph& graph, bool no_grad = true) override {
        assert(graph.num_nodes > 0);
        if (no_grad) {
            torch::NoGradGuard guard;
            torch::Tensor x = torch::ones({graph.num_nodes, 1}).to(cfg::device);
            torch::Tensor y = gnn->forward(x, graph.to_adj_tensor());
            torch::Tensor policy_pred = y.narrow(1, 0, 1).reshape({-1});
            policy_pred = policy_pred.softmax(0);
            torch::Tensor value_pred = y.narrow(1, 1, 1).reshape({-1});
            value_pred = normalize(value_pred);
            return {policy_pred, value_pred};
        } else {
            torch::Tensor x = torch::ones({graph.num_nodes, 1}).to(cfg::device);
            torch::Tensor y = gnn->forward(x, graph.to_adj_tensor());
            torch::Tensor policy_pred = y.narrow(1, 0, 1).reshape({-1});
            policy_pred = policy_pred.softmax(0);
            torch::Tensor value_pred = y.narrow(1, 1, 1).reshape({-1});
            value_pred = normalize(value_pred);
            return {policy_pred, value_pred};
        }
    }

    float train(const std::vector<int>& idxes) override {
        auto& graphs = TrainData.graphs;
        auto& actions = TrainData.actions;
        auto& policy_target = TrainData.pis;
        auto& value_target = TrainData.rewards;
        optimizer.zero_grad();
        torch::Tensor loss = torch::tensor(0.0).to(cfg::device);
        for (int idx : idxes) {
            torch::Tensor policy, value;
            std::tie(policy, value) = infer_one(graphs[idx], false);
            // policy
            torch::Tensor policy_target_tensor = from_float32_vector(policy_target[idx]);
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
