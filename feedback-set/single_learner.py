import sys
import os
import time
import glob
import numpy as np
from model_updater import has_best, lock, unlock
from train_controller import get_train_data_files

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
        print('[learner] best model not found', flush=True)

    for _ in range(int(opt['save_interval'])):
        api.ClearTrainData()

        train_data_files = []
        while 1:
            while lock(opt, 'data_all'):
                pass
            train_data_files = get_train_data_files(opt)
            if len(train_data_files) < int(opt['batch_num']):
                unlock(opt, 'data_all')
                print('[learner] training data not found', flush=True)
                time.sleep(np.random.randint(30, 120))
            else:
                break

        batch_num = int(opt['batch_num'])
        train_data_files = np.random.choice(train_data_files, batch_num)

        for train_data_file in train_data_files:
            api.AddTrainData(train_data_file)
        unlock(opt, 'data_all')

        api.Train()

    filename = 'model{}'.format(time.time())
    api.SaveModel(filename)
    os.system('touch {}/timestamp_{}'.format(save_dir, filename))
    print('[learner] generated new model', flush=True)
