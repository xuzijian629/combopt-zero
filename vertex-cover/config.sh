g_type=erdos_renyi
gnn_type=s2v

# test (ignored in training)
graph_name=er100_15
add_noise_test=1
test_type=greedy

min_n=15
max_n=20
p=0.5

batch_size=16
batch_num=20

# number of steps for temp=1
initial_phase_len=100

# parallel
num_parallel_generator=10
num_parallel_learner=4

# ucb
alpha=1.5

# normalize
use_sigmoid=0
beta=1.0

# noise
dirichlet_alpha=0.03
dirichlet_eps=0.25

# sample mean and std
num_play=20

# node hash
max_n_log=10

# rollout times / number of nodes
rollout_coef=3

# optimizer
learning_rate=0.001
weight_decay=0.0001

# s2v
s2v_embed_dim=64
s2v_reg_hidden=64
s2v_iter=5

# gin
gin_hidden_dim=32
gin_layer_num=5
gin_mlp_hidden_dim=16
gin_mlp_layer_num=5

# pgnn
pgnn_hidden_dim=8
pgnn_layer_num=2
pgnn_mlp_hidden_dim=8
pgnn_mlp_layer_num=2

# gcn
gcn_hidden_dim=32
gcn_layer_num=5

# gat
gat_hidden_num=32
gat_head_num=4
gat_layer_num=5
gat_leakyrelu_alpha=0.2

# model updater
wait_update=1
num_test_graph=50

# controller
cleanup_interval=1
keep_time=3

# learner
save_interval=15

dir_id=p-$p-bsize-$batch_size-bnum-$batch_num-hot-$initial_phase_len-parallel-$num_parallel_generator-$num_parallel_learner-alpha-$alpha-play-$num_play-rollout-$min_rollout-$max_rollout-$rollout_coef-lr-$learning_rate-l2-$weight_decay-save-$save_interval-cleanup-$cleanup_interval-keep-$keep_time

save_dir=$result_root/$dir_id
