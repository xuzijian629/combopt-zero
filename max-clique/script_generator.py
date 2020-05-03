import subprocess


def get_lines():
    with open('config.sh') as f:
        data = f.read().split('\n')
        return data


def get_args(lines):
    opt = {}
    for line in lines:
        if len(line) == 0 or line[0] == '#' or '=' not in line:
            continue
        args = line.split('=')
        assert len(args) == 2
        opt[args[0]] = args[1]
    return opt


def opt_to_string(opt):
    ret = ''
    for key in opt:
        ret += '    -{} ${} \\\n'.format(key, key)
    return ret


if __name__ == '__main__':
    script_id = input('Enter script_id\n')
    lines = get_lines()
    opt = get_args(lines)

    train_script = """#!/bin/bash
result_root=$(dirname "$0")/results/""" + script_id + """
""" + '\n'.join(lines) + """
if [ ! -e $save_dir ];
then
    mkdir -p $save_dir/data
fi

PYTHONPATH=. python $(dirname "$0")/train_controller.py \\
"""

    eval_script = """#!/bin/bash
result_root=$(dirname "$0")/results/""" + script_id + """
""" + '\n'.join(lines) + """
graph_path=$(dirname "$0")/../test_graphs/$graph_name

model_path=$save_dir/best
eval_save_dir=$(dirname "$0")/eval/$dir_id/$graph_name

if [ ! -e $eval_save_dir ];
then
    mkdir -p $eval_save_dir
fi

PYTHONPATH=. python $(dirname "$0")/evaluate.py \\
    -graph_path $graph_path \\
    -test_type $test_type \\
    -model_path $model_path \\
"""

    train_script += opt_to_string(opt)
    eval_script += opt_to_string(opt)

    train_script += '    2>&1 | tee -a -i $save_dir/log.txt\n'
    eval_script += '    2>&1 | tee -a -i $eval_save_dir/log.txt\n'

    with open('t_{}.sh'.format(script_id), 'w') as f:
        f.write(train_script)

    with open('e_{}.sh'.format(script_id), 'w') as f:
        f.write(eval_script)

    subprocess.getoutput('chmod +x t_{}.sh'.format(script_id))
    subprocess.getoutput('chmod +x e_{}.sh'.format(script_id))
