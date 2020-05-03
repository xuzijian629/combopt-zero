import os
import sys
import glob
import signal
import numpy as np
import subprocess
import time
import re
import atexit
from model_updater import lock, unlock


def escape_ansi(line):
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', line)


def get_pids(command):
    ret = escape_ansi(subprocess.getoutput(
        'ps aux | grep {}'.format(command))).split('\n')
    ret = list(map(lambda x: x.split()[1], ret))
    return ret


def kill(command):
    pids = get_pids(command)
    for pid in pids:
        subprocess.getoutput('kill -9 {}'.format(pid))


def get_running_generator_num():
    return int(subprocess.getoutput('ps aux | grep single_data_generator.py | wc -l')) - 2


def get_running_learner_num():
    return int(subprocess.getoutput('ps aux | grep single_learner.py | wc -l')) - 2


def get_gpu_usage():
    ret = subprocess.getoutput('nvidia-smi | grep %')
    ret = escape_ansi(ret).split('\n')
    if len(ret) == 1:
        return [0]
    ret = list(map(lambda x: int(x.split()[-3][:-1]), ret))
    return ret


def get_cpu_usage():
    return subprocess.getoutput('top -b -n 1 | head -n 20')


def start_single_data_generator():
    gpu_id = np.argmin(get_gpu_usage())
    os.system(
        'PYTHONPATH=. python single_data_generator.py -gpu_id {} '.format(gpu_id) + ' '.join(sys.argv[1:]) + ' &')
    time.sleep(1)


def start_single_learner():
    gpu_id = np.argmin(get_gpu_usage())
    os.system(
        'PYTHONPATH=. python single_learner.py -gpu_id {} '.format(gpu_id) + ' '.join(sys.argv[1:]) + ' &')
    time.sleep(1)


def get_train_data_files(opt):
    save_dir = opt['save_dir']
    data = glob.glob('{}/data/timestamp_train_data*'.format(save_dir))
    if len(data) == 0:
        return []
    return list(map(lambda x: x.split('/')[-1][10:], data))


def cleanup_data(opt):
    now = time.time()
    keep_time = int(opt['keep_time'])
    while lock(opt, 'data_all'):
        pass
    train_data_files = get_train_data_files(opt)
    if len(train_data_files) == 0:
        unlock(opt, 'data_all')
        return
    files_to_erase = list(filter(lambda x: float(
        x[10:]) < now - 60 * keep_time, train_data_files))
    for f in files_to_erase:
        subprocess.getoutput('rm {}/data/{}'.format(opt['save_dir'], f))
        subprocess.getoutput(
            'rm {}/data/timestamp_{}'.format(opt['save_dir'], f))
    unlock(opt, 'data_all')
    print('[controller] erased {}/{} old training data'.format(
        len(files_to_erase), len(train_data_files)), flush=True)


def on_exit():
    kill('model_updater.py')
    print('[controller] killed updater', flush=True)
    time.sleep(1)
    kill('single_data_generator.py')
    print('[controller] killed generator', flush=True)
    time.sleep(1)
    kill('single_learner.py')
    print('[controller] killed learner', flush=True)


def launch_updater():
    os.system(
        'PYTHONPATH=. python model_updater.py -gpu_id 0 ' + ' '.join(sys.argv[1:]) + ' &')
    os.system(
        'PYTHONPATH=. python model_updater.py -gpu_id 1 ' + ' '.join(sys.argv[1:]) + ' &')


if __name__ == '__main__':
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    num_parallel_generator = int(opt['num_parallel_generator'])
    num_parallel_learner = int(opt['num_parallel_learner'])
    save_dir = opt['save_dir']

    atexit.register(on_exit)

    start = time.time()
    log_interval = 10
    prev_log = start
    cleanup_interval = int(opt['cleanup_interval'])
    prev_cleanup = start
    prev_data_cnt = 0
    prev_learner_cnt = 0
    data_cnt = 0
    learner_cnt = 0
    updater_launched = False

    while 1:
        now = time.time()
        if now > prev_log + log_interval * 60:
            print('[controller]', get_cpu_usage())
            print('[controller]', 'GPU usage: {}'.format(
                get_gpu_usage()), flush=True)
            prev_log = now
        if now > prev_cleanup + cleanup_interval * 60:
            print('[controller] {} new data batch generated'.format(
                data_cnt - prev_data_cnt))
            print('[controller] {} batches used for training'.format(
                (learner_cnt - prev_learner_cnt) * int(opt['batch_num'])), flush=True)
            prev_data_cnt = data_cnt
            prev_learner_cnt = learner_cnt
            cleanup_data(opt)
            prev_cleanup = now

        if not updater_launched and now > start + int(opt['wait_update']) * 60:
            launch_updater()
            updater_launched = True

        if get_running_generator_num() < num_parallel_generator:
            start_single_data_generator()
            data_cnt += 1
        if get_running_learner_num() < num_parallel_learner:
            start_single_learner()
            learner_cnt += 1
