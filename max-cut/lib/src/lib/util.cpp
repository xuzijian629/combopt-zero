#include <torch/torch.h>

#include <cassert>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>

#include "config.h"
#include "graph.h"
#include "train_batch.h"

std::random_device rnd;

int weighted_choose(const std::vector<float>& prob) {
    float sum = 0;
    float border = (float)rnd() / std::random_device::max();
    for (int i = 0; i < (int)prob.size(); i++) {
        sum += prob[i];
        if (sum > border) return i;
    }
    assert(false);
}

torch::Tensor normalize(torch::Tensor t) {
    assert(t.sizes().size() == 1);
    auto var = t.pow(2).mean() - t.mean().pow(2) + cfg::eps;
    /*
    even we set cfg::eps = 1e-3 by default, var can easily be negative, for example, when we input a complete graph,
    where all elements in `t` is large and same.
    */
    assert(var.item<float>() > 0);
    auto ret = (t - t.mean()) * var.rsqrt();
    if (cfg::use_sigmoid) {
        return torch::tanh(ret / cfg::beta);
    } else {
        return ret;
    }
}

torch::Tensor from_float32_array(void* arr, torch::IntArrayRef sizes) {
    torch::TensorOptions option(torch::TensorOptions().dtype(torch::kFloat32));
    return torch::from_blob(arr, sizes, option).clone().to(cfg::device);
}
torch::Tensor from_float32_vector(std::vector<float> v) {
    return from_float32_array(&(*(v.begin())), {(int64_t)v.size()});
}

torch::Tensor cross_entropy(torch::Tensor input, torch::Tensor target) {
    return -(target * (input + cfg::eps).log()).sum();
}

void save_train_data(const std::string& filename, const TrainBatch& batch) {
    std::string path = cfg::save_dir + "data/" + filename;
    std::ofstream f(path);
    f << batch.n << '\n';
    for (int i = 0; i < batch.n; i++) {
        f << batch.graphs[i].num_nodes << ' ' << batch.graphs[i].num_edges << '\n';
        for (auto& p : batch.graphs[i].edge_list) {
            int u = p.first, v = p.second;
            f << u << ' ' << v << '\n';
        }
        assert((int)batch.adj_blacks[i].size() == batch.graphs[i].num_nodes);
        for (int a : batch.adj_blacks[i]) {
            f << a << ' ';
        }
        f << '\n';
        assert((int)batch.adj_whites[i].size() == batch.graphs[i].num_nodes);
        for (int a : batch.adj_whites[i]) {
            f << a << ' ';
        }
        f << '\n';
        f << batch.actions[i] << '\n';
        assert((int)batch.pis[i].size() == batch.graphs[i].num_nodes * 2);
        for (float a : batch.pis[i]) {
            f << std::setprecision(10) << a << ' ';
        }
        f << '\n';
        f << batch.rewards[i] << '\n';
    }
    f.close();
}

TrainBatch load_train_data(const std::string& filename) {
    std::string path = cfg::save_dir + "data/" + filename;
    std::ifstream f(path);
    int batch_n;
    f >> batch_n;
    std::vector<Graph> graphs(batch_n);
    std::vector<std::vector<int>> adj_blacks(batch_n), adj_whites(batch_n);
    std::vector<int> actions(batch_n);
    std::vector<std::vector<float>> pis(batch_n);
    std::vector<float> rewards(batch_n);
    for (int i = 0; i < batch_n; i++) {
        int n, m;
        f >> n >> m;
        Graph g(n);
        for (int j = 0; j < m; j++) {
            int u, v;
            f >> u >> v;
            g.add_edge(u, v);
        }
        graphs[i] = g;
        adj_blacks[i].resize(n);
        for (int j = 0; j < n; j++) f >> adj_blacks[i][j];
        adj_whites[i].resize(n);
        for (int j = 0; j < n; j++) f >> adj_whites[i][j];
        f >> actions[i];
        pis[i].resize(n * 2);
        for (int j = 0; j < n * 2; j++) {
            f >> pis[i][j];
        }
        f >> rewards[i];
    }
    f.close();
    return TrainBatch(graphs, adj_blacks, adj_whites, actions, pis, rewards);
}
