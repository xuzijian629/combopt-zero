import sys
import networkx as nx
from lib.lib import FeedbackSetLib
from model_updater import has_best


def get_graph_from_path(path):
    g = nx.Graph()
    with open(path) as f:
        data = f.read().split('\n')
        n, m = map(int, data[0].split())
        for i in range(m):
            u, v = map(int, data[1 + i].split())
            g.add_edge(u, v)
    return g


if __name__ == '__main__':
    api = FeedbackSetLib(sys.argv)

    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    assert has_best(opt)
    api.LoadModel('best')
    api.SetCurrentTestGraph(get_graph_from_path(opt['graph_path']))

    print('[evaluate] model_path: {}'.format(opt['model_path']))
    if opt['test_type'] == 'greedy':
        print('[evaluate] evaluated {} by greedy: {}'.format(
            opt['graph_name'], api.Test()), flush=True)
    elif opt['test_type'] == 'mcts':
        print('[evaluate] evaluated {} by mcts: {}'.format(
            opt['graph_name'], api.TestByMCTS()), flush=True)
