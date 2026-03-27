#!/bin/bash
set -e

mode=${1:-quick}

if [ "$mode" = "full" ]; then
    eval_samples=0
    echo "=============== Running full data evaluation ==============="
else
    eval_samples=0
    echo "=============== Running quick evaluation ==============="
fi

export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false
export TORCHCODEC_NUM_THREADS=32
export TORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/this_does_not_exist
export CUDA_MPS_LOG_DIRECTORY=/tmp/this_does_not_exist

# check DS_API_KEY is set
if [ -z "$DS_API_KEY" ]; then
    echo "DS_API_KEY is not set"
    exit 1
fi

tasks=needle
model_name=Qwen3-8B
seq_len=32768


# shadowkv-t
lr_att_mode=shadowkv-60-16
lr_ratio=none
token_group=none
budget=400
start_layer=none
./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

# loki-t
loki_lr_ratio=0.03125
lr_att_mode=loki
budget=400
start_layer=0-curr-emb
loki_token_group=1
./scripts/eval.sh $tasks $model_name $lr_att_mode $loki_lr_ratio $loki_token_group $budget $start_layer $eval_samples $seq_len


# kvswap-t-nvme
lr_ratio=0.25
lr_att_mode=lr_proj_mh
start_layer=0-curr-emb
budget=400
token_group=4
./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len


if [ -z "${EVAL_USER:-}" ] || [ "${EVAL_USER}" = '$EVAL_USER' ]; then
  echo "EVAL_USER is not correctly set, EVAL_USER=$EVAL_USER. Exit."
  exit 1
fi

# plot
shadowkv_t_result=./exps/${EVAL_USER}/results/needle/Qwen3-8B_32768/shadowkv-60-16/b${budget}-DeepSeek_deepseek-chat.jsonl
loki_t_result=./exps/${EVAL_USER}/results/needle/Qwen3-8B_32768/loki/b${budget}_p${loki_lr_ratio}_${start_layer}_tg${loki_token_group}-DeepSeek_deepseek-chat.jsonl
kvswap_t_result=./exps/${EVAL_USER}/results/needle/Qwen3-8B_32768/lr_proj_mh/b${budget}_p${lr_ratio}_${start_layer}_tg${token_group}-DeepSeek_deepseek-chat.jsonl

mkdir -p ./RESULTS/${EVAL_USER}
output_png=./RESULTS/${EVAL_USER}/fig-9.png

echo "Plotting figure 9..."

source .venv/bin/activate
python bench/Needle_test/plot_all.py --shadowkv-t-result $shadowkv_t_result \
--loki-t-result $loki_t_result \
--kvswap-t-result $kvswap_t_result \
--output-pdf $output_png > ${output_png}.log 2>&1

if [ $? -ne 0 ]; then
    echo "Failed to plot figure 9"
    exit 1
fi

echo "Figure 9 has been saved to $output_png"

