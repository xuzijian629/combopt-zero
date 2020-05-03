#include "libtest.h"

#include "env.h"
#include "graph.h"
#include "hash.h"
#include "mcts.h"
#include "util.h"

Graph generate_random_graph() {
    int n = rnd() % 20 + 1;
    Graph g(n);
    std::set<std::pair<int, int>> edges;
    for (int i = 0; i < std::max(n * 3, n * (n - 1) / 2); i++) {
        int a = rnd() % n, b = rnd() % n;
        if (a == b) continue;
        if (edges.count(std::minmax(a, b)) == 0) {
            edges.insert(std::minmax(a, b));
            g.add_edge(a, b);
        }
    }
    return g;
}

void test_sparse() {
    for (int i = 0; i < 1000; i++) {
        Graph g = generate_random_graph();
        assert(get_adj_hash(g) == get_adj_hash_dense(g));
    }
}

void test_random_play() {
    Graph g = generate_random_graph();
    const int times = 10000;
    std::vector<int> random_play_fast(times), random_play_by_env(times);
    for (int i = 0; i < times; i++) {
        random_play_fast[i] = random_play(g);
    }
    for (int i = 0; i < times; i++) {
        int reward = 0;
        Graph g_ = g;
        while (!is_end(g_)) {
            reward++;
            g_ = step(g_, rnd() % g_.num_nodes);
        }
        random_play_by_env[i] = reward;
    }
    double fast_mean = std::accumulate(random_play_fast.begin(), random_play_fast.end(), 0.0) / 1000;
    double env_mean = std::accumulate(random_play_by_env.begin(), random_play_by_env.end(), 0.0) / 1000;
    if (abs(fast_mean - env_mean) >= 1) {
        std::cerr << "fast mean: " << fast_mean << ", env_mean: " << env_mean << std::endl;
        assert(abs(fast_mean - env_mean) < 1);
    }
}

void test_all() {
    test_sparse();
    test_random_play();
}
