import sys
import os
import glob
import time
import subprocess
import numpy as np
import networkx as nx

from lib.lib import FeedbackSetLib


def gen_graph(opt, test=False):
    max_n = int(opt['max_n'])
    min_n = max_n if test else int(opt['min_n'])
    n = np.random.randint(min_n, max_n + 1)
    if opt['g_type'] == 'erdos_renyi':
        return nx.erdos_renyi_graph(n=n, p=float(opt['p']))
    raise AssertionError('Unknown graph type: {}'.format(opt['g_type']))

# returns 0 when successfully locked


def lock(opt, filename):
    return os.system('mkdir {}/lock_{} &> /dev/null'.format(opt['save_dir'], filename))


def unlock(opt, filename):
    os.system('rmdir {}/lock_{}'.format(opt['save_dir'], filename))


def update_best(opt, filename):
    while lock(opt, 'best'):
        pass
    save_dir = opt['save_dir']
    os.system('cp {}/{} {}/best'.format(save_dir, filename, save_dir))
    unlock(opt, 'best')


def has_best(opt):
    save_dir = opt['save_dir']
    best = glob.glob(save_dir + '/best')
    return len(best) == 1


def has_timestamp(opt, filename):
    return len(glob.glob('{}/timestamp_{}'.format(opt['save_dir'], filename))) == 1


if __name__ == '__main__':
    print('[updater] started', flush=True)
    api = FeedbackSetLib(sys.argv)

    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    save_dir = opt['save_dir']

    start = time.time()

    # when to save next [min]
    next_save = 5

    while 1:
        now = time.time()
        elapsed = (now - start) / 60
        if opt['gpu_id'] == '0' and elapsed > next_save and has_best(opt):
            while lock(opt, 'best'):
                pass
            os.system(
                'cp {}/best {}/best_{}'.format(save_dir, save_dir, next_save))
            next_save += 5
            unlock(opt, 'best')

        models = glob.glob(save_dir + '/timestamp_*')
        if len(models) == 0:
            continue
        filename = sorted(models, key=lambda x: float(
            x.split('/')[-1][15:]))[-1].split('/')[-1][10:]
        if lock(opt, 'timestamp_{}'.format(filename)):
            continue
        # check if file exists
        if not has_timestamp(opt, filename):
            unlock(opt, 'timestamp_{}'.format(filename))
            continue
        print('[updater] loading', filename, flush=True)
        api.LoadModel(filename)
        subprocess.getoutput('rm {}/timestamp_{}'.format(save_dir, filename))
        unlock(opt, 'timestamp_{}'.format(filename))

        test_graphs = []
        for i in range(int(opt['num_test_graph'])):
            test_graphs.append(gen_graph(opt, True))
        score = 0
        for graph in test_graphs:
            api.SetCurrentTestGraph(graph)
            score += api.Test()

        best_score = 0
        if has_best(opt):
            while lock(opt, 'best'):
                pass
            api.LoadModel('best')
            unlock(opt, 'best')
            for graph in test_graphs:
                api.SetCurrentTestGraph(graph)
                best_score += api.Test()
        else:
            best_score = 10000000000000

        print('[updater] {} vs {}'.format(score, best_score), flush=True)
        if score < best_score or (score == best_score and np.random.randint(0, 2)):
            update_best(opt, filename)
            print('[updater] best updated!', flush=True)
        os.system('rm {}/{}'.format(save_dir, filename))
