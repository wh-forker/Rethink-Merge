#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=VGG19_cifar10_loop10

# cifar10
start=1
end=10
for i in $(seq $start $end); do
    model_name="${name}_${i}"
    CUDA_VISIBLE_DEVICES=$1 python -u train_cifar10.py --name ${model_name} --model vgg19 > logs/${time}_train_${model_name}.log 2>&1 &
done
