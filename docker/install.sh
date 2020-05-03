#!/bin/bash
set -e


base_dir=$(dirname $0)/..
abs_base_dir=$(cd $base_dir && pwd -P)

source $base_dir/docker/config.sh


# Download LibTorch (for Linux)
mkdir -p $base_dir/docker/workspace
echo 'Downloading LibTorch from pytorch.org...'
libtorch_path=$abs_base_dir/docker/workspace/libtorch

if [ -f $libtorch_path/bin ]; then
    echo 'skipped'
else
    # Using newer version may cause errors
    libtorch_url='https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.3.0%2Bcpu.zip'
    curl $libtorch_url -o $libtorch_path.zip
    unzip $libtorch_path.zip -d $(dirname $libtorch_path)
fi


# Build
lib_path=$abs_base_dir/$problem/lib
if [ ! -f $lib_path/CMakeLists.txt ]; then
    echo 'Invalid library path:' $lib_path
    echo 'Please check problem name:' $problem
    exit 1
fi
docker run -v $abs_base_dir:$abs_base_dir python:3.7 bash -c "\
    cd $abs_base_dir/$problem/lib && \
    rm -rf build && \
    mkdir build && \
    cd build && \
    apt update && \
    apt install cmake -y && \
    cmake -DCMAKE_PREFIX_PATH=$libtorch_path .. && \
    make"
