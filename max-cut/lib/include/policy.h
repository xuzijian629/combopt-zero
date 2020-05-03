#ifndef POLICY_H
#define POLICY_H

#include <torch/torch.h>

#include <memory>
#include <vector>

#include "graph.h"

class Policy {
public:
    Policy();
    virtual ~Policy() = 0;

    // 特徴量なし！ｗ
    virtual std::pair<torch::Tensor, torch::Tensor> infer_one(const Graph& graph, const std::vector<int>& adj_black,
                                                              const std::vector<int>& adj_white,
                                                              bool no_grad = true) = 0;

    // return loss
    virtual float train(const std::vector<int>& idxes) = 0;

    virtual void save(const std::string& filename) = 0;
    virtual void load(const std::string& filename) = 0;
};

extern std::shared_ptr<Policy> global_policy;

#endif
