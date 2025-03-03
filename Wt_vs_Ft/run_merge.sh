#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=vgg19_x100_Merge
arch=vgg19
script=cifar10_merge.py
magnitude=100

CUDA_VISIBLE_DEVICES=$1 python -u ${script} --model ${arch} --model_num 2 --magnitude ${magnitude} > logs/${time}_${name}_2models.log 2>&1 &
CUDA_VISIBLE_DEVICES=$1 python -u ${script} --model ${arch} --model_num 3 --magnitude ${magnitude} > logs/${time}_${name}_3models.log 2>&1 &
CUDA_VISIBLE_DEVICES=$1 python -u ${script} --model ${arch} --model_num 5 --magnitude ${magnitude} > logs/${time}_${name}_5models.log 2>&1 &
CUDA_VISIBLE_DEVICES=$1 python -u ${script} --model ${arch} --model_num 7 --magnitude ${magnitude} > logs/${time}_${name}_7models.log 2>&1 &
CUDA_VISIBLE_DEVICES=$1 python -u ${script} --model ${arch} --model_num 10 --magnitude ${magnitude} > logs/${time}_${name}_10models.log 2>&1 &
