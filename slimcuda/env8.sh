#!/bin/sh

export LD_LIBRARY_PATH=/opt/nvidia8/cuda/lib64
export PATH=/opt/nvidia8/cuda/bin:$PATH
export NVIDIA_SDK_VERSION=8
export EXTRA_CFLAGS="-Wl,-rpath -Wl,/opt/nvidia8/cuda/lib64"
export EXTRA_NVCC_CFLAGS='-Xlinker "-rpath /opt/nvidia8/cuda/lib64"'
export NVIDIA_SDK_VERSION=8

