#!/bin/bash
set -e


base_dir=$(dirname $0)/..
abs_base_dir=$(cd $base_dir && pwd -P)

source $base_dir/docker/config.sh


# Run Training
echo 'You can stop training anytime by terminating docker process'
docker run -v $abs_base_dir:$abs_base_dir python:3.7 timeout $time_limit_sec bash -c "\
    cd $abs_base_dir/$problem && \
    echo sample | python script_generator.py && \
    pip install -r $abs_base_dir/requirements.txt && \
    ./t_sample.sh"
