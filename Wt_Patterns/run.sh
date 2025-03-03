#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=cifar100_linear_mixup

#CUDA_VISIBLE_DEVICES=$1 python -u Linear_vis_cifar100.py > logs/${time}_train_${name}.log 2>&1 &
CUDA_VISIBLE_DEVICES=$1 python -u Linear_vis_cifar100_mixup.py > logs/${time}_train_${name}.log 2>&1 &
