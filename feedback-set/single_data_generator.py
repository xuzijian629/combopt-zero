import sys
import os
import time
import glob
from model_updater import gen_graph, has_best, lock, unlock

from lib.lib import FeedbackSetLib


if __name__ == '__main__':
    api = FeedbackSetLib(sys.argv)

    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    save_dir = opt['save_dir']
    if has_best(opt):
        while lock(opt, 'best'):
            pass
        api.LoadModel('best')
        unlock(opt, 'best')
    else:
        print('[generator] best model not found', flush=True)

    api.SetCurrentGraph(gen_graph(opt))
    filename = 'train_data{}'.format(time.time())
    api.GenerateTrainData(filename)
    os.system('touch {}/data/timestamp_{}'.format(save_dir, filename))
    print('[generator] generated new data', flush=True)
