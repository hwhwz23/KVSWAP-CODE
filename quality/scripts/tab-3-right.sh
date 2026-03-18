#!/bin/bash
set -e

mode=${1:-quick}

if [ "$mode" = "full" ]; then
    eval_samples=0
    echo "=============== Running full data evaluation ==============="
else
    eval_samples=50
    echo "=============== Running quick evaluation ==============="
fi

export PYTHONPATH=$(pwd)/src
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

tasks=mlvu
seq_len=32768

func(){
    # full-kv
    lr_att_mode=none
    lr_ratio=none
    token_group=none
    budget=none
    start_layer=none
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

    # loki
    loki_lr_ratio=${loki_ratio}
    lr_att_mode=loki
    budget=400
    start_layer=0-curr-emb
    loki_token_group=1
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $loki_lr_ratio $loki_token_group $budget $start_layer $eval_samples $seq_len

    # shadowkv
    lr_att_mode=${shadowkv_mode}
    lr_ratio=none
    token_group=none
    budget=400
    start_layer=none
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

    # kvswap-nvme
    lr_ratio=${kvswap_ratio}
    lr_att_mode=lr_proj_mh
    budget=400
    start_layer=0-curr-emb
    token_group=4
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

    # kvswap-emmc
    lr_ratio=${kvswap_ratio}
    budget=400
    lr_att_mode=lr_proj_mh
    start_layer=0-curr-emb
    token_group=8
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

    # loki-t
    loki_lr_ratio=${loki_t_ratio}
    lr_att_mode=loki
    budget=400
    start_layer=0-curr-emb
    loki_token_group=1
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $loki_lr_ratio $loki_token_group $budget $start_layer $eval_samples $seq_len

    # shadowkv-t
    lr_att_mode=${shadowkv_t_mode}
    lr_ratio=none
    token_group=none
    budget=400
    start_layer=none
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

    # kvswap-t-nvme
    lr_ratio=${kvswap_t_ratio}
    lr_att_mode=lr_proj_mh
    budget=400
    start_layer=0-curr-emb
    token_group=4
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

    # kvswap-t-emmc
    lr_ratio=${kvswap_t_ratio}
    budget=400
    lr_att_mode=lr_proj_mh
    start_layer=0-curr-emb
    token_group=8
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

}

###############################################################
model_name=Qwen2.5-VL-3B-Instruct
shadowkv_mode=shadowkv-32-24
shadowkv_t_mode=shadowkv-80-6
loki_ratio=0.125
loki_t_ratio=0.03125
kvswap_ratio=0.25
kvswap_t_ratio=0.0625
# prepare_mlvu
./scripts/eval.sh prepare_mlvu $model_name none none none none none $eval_samples $seq_len
# run 
func


####################################################################
model_name=Qwen2.5-VL-7B-Instruct
shadowkv_mode=shadowkv-32-48
shadowkv_t_mode=shadowkv-80-10
loki_ratio=0.125
loki_t_ratio=0.03125
kvswap_ratio=0.5
kvswap_t_ratio=0.125
# prepare_mlvu
./scripts/eval.sh prepare_mlvu $model_name none none none none none $eval_samples $seq_len
# run 
func


####################################################################
model_name=InternVL3-14B
shadowkv_mode=shadowkv-16-40-48-4
shadowkv_t_mode=shadowkv-60-16
loki_ratio=0.125
loki_t_ratio=0.03125
kvswap_ratio=1
kvswap_t_ratio=0.25
func


####################################################################


mkdir -p ./RESULTS

if [ "$mode" = "full" ]; then
    output_file=./RESULTS/tab-3-right-full.txt
else
    output_file=./RESULTS/tab-3-right.txt
fi

python ./scripts/utils.py ./exps/results/mlvu/{model_name}_${seq_len} table3-right | tee $output_file


