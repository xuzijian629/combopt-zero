#!/bin/bash
set -e


base_dir=$(dirname $0)/..
abs_base_dir=$(cd $base_dir && pwd -P)

source $base_dir/docker/config.sh


docker run -v $abs_base_dir:$abs_base_dir python:3.7 timeout $time_limit_sec bash -c "\
    cd $abs_base_dir/$problem && \
    pip install -r $abs_base_dir/requirements.txt && \
    ./e_sample.sh"
