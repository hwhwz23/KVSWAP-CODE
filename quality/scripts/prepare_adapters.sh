#!/bin/bash


MODEL=Qwen3-4B

######################################################
python ./src/prepare_adapter.py \
    --model_path ./MODELS/${MODEL} \
    --seq_len 32768 \
    --ratios 0.125 \
    --mode infinigen \
    --save_dir ./new_adapters \
    --eval_samples 20 \
    --task c4

######################################################
######################################################
python ./src/prepare_adapter.py \
    --model_path ./MODELS/${MODEL} \
    --seq_len 32768 \
    --ratios 0.125 \
    --mode loki \
    --save_dir ./new_adapters \
    --eval_samples 20 \
    --task c4


######################################################
python ./src/prepare_adapter.py \
    --model_path ./MODELS/${MODEL} \
    --seq_len 32768 \
    --ratios 1 \
    --mode kvswap \
    --save_dir ./new_adapters \
    --eval_samples 20 \
    --task c4

######################################################