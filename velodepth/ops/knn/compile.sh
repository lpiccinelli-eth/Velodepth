#!/usr/bin/env bash

export TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX" 
# export FORCE_CUDA=1 #if you do not actually have cuda, workaround
python setup.py build install