#ifndef POLICIES_H
#define POLICIES_H

#include "gat.h"
#include "gcn.h"
#include "gin.h"
#include "gnn_policy.h"
#include "pgnn.h"
#include "s2v.h"

using S2VPolicy = GNNPolicy<S2V>;
using GINPolicy = GNNPolicy<GraphIsomorphismNetwork>;
using PGNNPolicy = GNNPolicy<PGNN>;
using GCNPolicy = GNNPolicy<GraphConvolutionalNetwork>;
using GATPolicy = GNNPolicy<GraphAttentionNetwork>;

#endif
