#ifndef LIB_H
#define LIB_H

extern "C" void Init(const int argc, const char **argv);

extern "C" void SetCurrentGraph(int num_nodes, int num_edges, const int *edges_from, const int *edges_to);

extern "C" void SetCurrentTestGraph(int num_nodes, int num_edges, const int *edges_from, const int *edges_to);

extern "C" void LoadModel(const char *filename, bool from_save_dir);

extern "C" void SaveModel(const char *filename);

extern "C" int Test();

extern "C" int TestByMCTS();

extern "C" void GenerateTrainData(const char *filename);

extern "C" void ClearTrainData();

extern "C" void AddTrainData(const char *filename);

extern "C" float Train();

#endif
