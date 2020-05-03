#ifndef MOCK_POLICY_H
#define MOCK_POLICY_H

#include "policy.h"

class MockPolicy : public Policy {
public:
    MockPolicy();
    std::pair<torch::Tensor, torch::Tensor> infer_one(const Graph& graph, bool no_grad) override;
    float train(const std::vector<int>& idxes) override;
    void save(const std::string& filename) override;
    void load(const std::string& filename) override;
};

#endif
