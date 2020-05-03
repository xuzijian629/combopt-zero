#include "lib.h"

#include <cassert>

#include "config.h"
#include "env.h"
#include "graph.h"
#include "hash.h"
#include "libtest.h"
#include "mcts.h"
#include "mock_policy.h"
#include "policies.h"
#include "policy.h"
#include "train_batch.h"

void Init(const int argc, const char** argv) {
    cfg::LoadParams(argc, argv);
    init_hash();
    // test_all();
    if (cfg::gnn_type == "s2v") {
        global_policy = std::make_shared<S2VPolicy>();
    } else if (cfg::gnn_type == "gin") {
        global_policy = std::make_shared<GINPolicy>();
    } else if (cfg::gnn_type == "gcn") {
        global_policy = std::make_shared<GCNPolicy>();
    } else if (cfg::gnn_type == "gat") {
        global_policy = std::make_shared<GATPolicy>();
    } else if (cfg::gnn_type == "pgnn") {
        global_policy = std::make_shared<PGNNPolicy>();
    } else {
        assert(false);
    }
}

void SetCurrentGraph(int num_nodes, int num_edges, const int* edges_from, const int* edges_to) {
    Graph g(num_nodes);
    for (int i = 0; i < num_edges; i++) {
        g.add_edge(edges_from[i], edges_to[i]);
    }
    CurrentGraph = g;
}

void SetCurrentTestGraph(int num_nodes, int num_edges, const int* edges_from, const int* edges_to) {
    Graph g(num_nodes);
    for (int i = 0; i < num_edges; i++) {
        g.add_edge(edges_from[i], edges_to[i]);
    }
    CurrentTestGraph = g;
}

void SaveModel(const char* filename) { global_policy->save(cfg::save_dir + std::string(filename)); }
void LoadModel(const char* filename, bool from_save_dir) {
    global_policy->load((from_save_dir ? cfg::save_dir : "") + std::string(filename));
}

int Test() { return test(); }

int TestByMCTS() { return test_by_mcts(); }

void ClearTrainData() { TrainData.clear(); }

void GenerateTrainData(const char* filename) { generate_train_data(std::string(filename)); }

void AddTrainData(const char* filename) {
    TrainBatch batch = load_train_data(std::string(filename));
    TrainData += batch;
}

float Train() { return train(); }
