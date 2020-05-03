#ifndef UTIL_H
#define UTIL_H

#include <torch/torch.h>

#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "graph.h"
#include "train_batch.h"

extern std::random_device rnd;

template <typename T>
std::pair<float, float> mean_std(const std::vector<T>& x) {
    int n = x.size();
    float sum = std::accumulate(x.begin(), x.end(), 0.0);
    std::vector<float> xx(n);
    for (int i = 0; i < n; i++) xx[i] = 1.0 * x[i] * x[i];
    float mean = sum / n;
    float var = (std::accumulate(xx.begin(), xx.end(), 0.0) - mean * mean) / n + cfg::eps;
    assert(var > 0);
    return std::make_pair(mean, std::sqrt(var));
}

template <typename T>
int argmax(const std::vector<T>& x) {
    assert(!x.empty());
    float max = x[0];
    int idx = 0;
    for (int i = 1; i < (int)x.size(); i++) {
        if (x[i] > max) {
            max = x[i];
            idx = i;
        }
    }
    return idx;
}

template <typename T>
T max(const std::vector<T>& x) {
    assert(!x.empty());
    T ret = x[0];
    for (int i = 1; i < (int)x.size(); i++) {
        if (x[i] > ret) ret = x[i];
    }
    return ret;
}

int weighted_choose(const std::vector<float>& prob);

// note: give 1D tensor
template <class T>
std::vector<T> tensor_to_vector(torch::Tensor t) {
    t = t.to(torch::kCPU);
    assert(t.sizes().size() == 1);
    t.contiguous();
    return std::vector<T>(t.data_ptr<T>(), t.data_ptr<T>() + t.numel());
}

// note: give 1D tensor
// make mean 0 and std 1
torch::Tensor normalize(torch::Tensor t);

torch::Tensor from_float32_array(void* arr, torch::IntArrayRef sizes);
torch::Tensor from_float32_vector(std::vector<float> v);

torch::Tensor cross_entropy(torch::Tensor input, torch::Tensor target);

void save_train_data(const std::string& filename, const TrainBatch& batch);
TrainBatch load_train_data(const std::string& filename);

#endif
