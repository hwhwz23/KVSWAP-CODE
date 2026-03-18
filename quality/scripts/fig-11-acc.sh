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


model_name=Llama-3.1-8B-Instruct
seq_len=32768

func(){
    # full-kv
    # lr_att_mode=none
    # lr_ratio=none
    # token_group=none
    # budget=none
    # start_layer=none
    # ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len
    # SKIP, can refer to some results produced by tab-2.sh

    # infinigen*
    lr_att_mode=infinigen_mergeh
    lr_ratio=0.5
    token_group=1
    budget=400
    start_layer=1
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

    # loki
    loki_lr_ratio=0.5
    lr_att_mode=loki
    budget=400
    start_layer=0-curr-emb
    loki_token_group=1
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $loki_lr_ratio $loki_token_group $budget $start_layer $eval_samples $seq_len

    # shadowkv
    lr_att_mode=shadowkv-8-160-48-4
    lr_ratio=none
    token_group=none
    budget=400
    start_layer=none
    ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len

    # kvswap-nvme
    # lr_ratio=1
    # lr_att_mode=lr_proj_mh
    # budget=400
    # start_layer=0-curr-emb
    # token_group=4
    # ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len
    # SKIP, can refer to some results produced by tab-2.sh

    # kvswap-emmc
    # lr_ratio=1
    # budget=400
    # lr_att_mode=lr_proj_mh
    # start_layer=0-curr-emb
    # token_group=8
    # ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len
    # SKIP, can refer to some results produced by tab-2.sh

    # kvswap-t-nvme
    # lr_ratio=0.25
    # lr_att_mode=lr_proj_mh
    # budget=400
    # start_layer=0-curr-emb
    # token_group=4
    # ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len
    # SKIP, can refer to some results produced by tab-2.sh

    # kvswap-t-emmc
    # lr_ratio=0.25
    # budget=400
    # lr_att_mode=lr_proj_mh
    # start_layer=0-curr-emb
    # token_group=8
    # ./scripts/eval.sh $tasks $model_name $lr_att_mode $lr_ratio $token_group $budget $start_layer $eval_samples $seq_len
    # SKIP, can refer to some results produced by tab-2.sh

}


tasks=longbench_mqa+sum
func


mkdir -p ./RESULTS

if [ "$mode" = "full" ]; then
    output_file=./RESULTS/fig-11-acc-full.txt
else
    output_file=./RESULTS/fig-11-acc.txt
fi

python ./scripts/utils.py ./exps/results/${tasks}/${model_name}_${seq_len} fig-11-acc | tee $output_file




