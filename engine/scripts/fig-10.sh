#!/bin/bash

run_mode=${1:-quick}

if [ "$run_mode" = "full" ]; then
    MAX_COUNT=40
    echo "=============== Running full evaluation ==============="
else
    MAX_COUNT=2
    echo "=============== Running quick evaluation ==============="
fi


clear_offload_dir(){
    disk_type=$1
    if [ "$disk_type" = "emmc" ]; then
      if [ -z "$EMMC_OFFLOAD_DIR" ]; then
        echo "EMMC_OFFLOAD_DIR is not set. Exit."
        exit 1
      fi
      if [ -z "$EMMC_DEV_NAME" ]; then
        echo "EMMC_DEV_NAME is not set. Exit."
        exit 1
      fi
      rm -rf $EMMC_OFFLOAD_DIR/*
      echo "Cleaned up $EMMC_OFFLOAD_DIR"
    elif [ "$disk_type" = "usb" ]; then
      if [ -z "$USB_OFFLOAD_DIR" ]; then
        echo "USB_OFFLOAD_DIR is not set. Exit."
        exit 1
      fi
      if [ -z "$USB_DEV_NAME" ]; then
        echo "USB_DEV_NAME is not set. Exit."
        exit 1
      fi
      rm -rf $USB_OFFLOAD_DIR/*
      echo "Cleaned up $USB_OFFLOAD_DIR"
    elif [ "$disk_type" = "nvme" ]; then
      if [ -z "$NVME_OFFLOAD_DIR" ]; then
        echo "NVME_OFFLOAD_DIR is not set. Exit."
        exit 1
      fi
      if [ -z "$NVME_DEV_NAME" ]; then
        echo "NVME_DEV_NAME is not set. Exit."
        exit 1
      fi
      rm -rf $NVME_OFFLOAD_DIR/*
      echo "Cleaned up $NVME_OFFLOAD_DIR"
    else
      echo "Invalid disk type $disk_type. Exit."
      exit 1
    fi
}


################ShadowKV#######################
run_shadowkv(){
    # chunk_size=8
    # rank=160
    chunk_size=16
    rank=40
    ./src/shadowkv/run_shadowkv.sh $TEST_MODEL $DISK_TYPE $TOTAL_LEN \
        $SEED $MAX_NUM_KV $chunk_size $rank "$BATCH_LIST"
}

################KVSwap#######################
run_kvswap(){
    RUN_ARGS=L4 
    USE_TOKEN_CACHE=0
    START_LAYER=0-curr-emb
    # USE_TOKEN_CACHE=1
    # START_LAYER=0-curr
    LR_PROJ_MODE=lr_proj_mh
    TOKEN_GROUP=$KVSWAP_TG
    MAX_NUM_KV=$MAX_KV_FETCH
    REUSE_BUDGET=$MAX_NUM_KV
    LR_PROJ_RATIO=$KVSWAP_RATIO
    ./scripts/eval.sh $TEST_MODEL $DISK_TYPE $LR_PROJ_MODE $TOTAL_LEN $TOKEN_GROUP \
        $MAX_NUM_KV $SEED $REUSE_BUDGET $LR_PROJ_RATIO $START_LAYER $USE_TOKEN_CACHE \
        $RUN_ARGS "$BATCH_LIST"
}



#############################################

MAX_KV_FETCH=400
KVSWAP_RATIO=1
SEQ_LIST=(32768)

run_model_run(){

  clear_offload_dir nvme
  clear_offload_dir emmc
  COUNT=0
  while IFS= read -r seed; do
      COUNT=$((COUNT + 1))
      for TOTAL_LEN in "${SEQ_LIST[@]}"; do
          echo "Running with sequence length: $TOTAL_LEN and seed: $seed"
          SEED=$seed
          #########################################################
          export MAX_ALLOC_KV_SIZE=$NVME_ALLOC_KV_SIZE
          BATCH_LIST="$NVME_BATCH_LIST"
          DISK_TYPE=nvme
          KVSWAP_TG=4
          run_kvswap
          #########################################################
          export MAX_ALLOC_KV_SIZE=$EMMC_ALLOC_KV_SIZE
          BATCH_LIST="$EMMC_BATCH_LIST"
          DISK_TYPE=emmc
          KVSWAP_TG=8
          run_kvswap
          #########################################################
      done
      if [ "$COUNT" -ge "$MAX_COUNT" ]; then
          break
      fi
  done < "./data/seeds.txt"


  #############################################

  clear_offload_dir nvme
  clear_offload_dir emmc
  COUNT=0
  while IFS= read -r seed; do
      COUNT=$((COUNT + 1))
      for TOTAL_LEN in "${SEQ_LIST[@]}"; do
          echo "Running with sequence length: $TOTAL_LEN and seed: $seed"
          SEED=$seed
          #########################################################
          export MAX_ALLOC_KV_SIZE=$NVME_ALLOC_KV_SIZE
          BATCH_LIST="$NVME_BATCH_LIST"
          DISK_TYPE=nvme
          run_shadowkv
          #########################################################
          export MAX_ALLOC_KV_SIZE=$EMMC_ALLOC_KV_SIZE
          BATCH_LIST="$EMMC_BATCH_LIST"
          DISK_TYPE=emmc
          run_shadowkv
          #########################################################
      done
      if [ "$COUNT" -ge "$MAX_COUNT" ]; then
          break
      fi
  done < "./data/seeds.txt"


  clear_offload_dir nvme
  clear_offload_dir emmc
  #############################################
  # run vLLM
  SEQ_LIST="32768"
  BATCH_LIST="1,8"

  ./scripts/run_vllm.sh $TEST_MODEL $SEQ_LIST $BATCH_LIST

  #############################################

}

NVME_ALLOC_KV_SIZE=$((1024*1024*2048))
EMMC_ALLOC_KV_SIZE=$((1024*1024*1024))
NVME_BATCH_LIST="1 8"
EMMC_BATCH_LIST="1 8"
TEST_MODEL=Llama-3.2-3B-Instruct
run_model_run



NVME_ALLOC_KV_SIZE=$((1024*1024*2048))
EMMC_ALLOC_KV_SIZE=$((1024*1024*1024))
NVME_BATCH_LIST="1 8"
EMMC_BATCH_LIST="1 8"
TEST_MODEL=Llama-3.1-8B-Instruct 
run_model_run


NVME_ALLOC_KV_SIZE=$((1024*1024*2048))
EMMC_ALLOC_KV_SIZE=$((1024*1024*768))
NVME_BATCH_LIST="1 8"
EMMC_BATCH_LIST="1"
TEST_MODEL=Qwen3-14B
run_model_run

#############################################
# Output Results
mkdir -p ./RESULTS

if [ -z "$EVAL_USER" ]; then
    echo "EVAL_USER is not set. Exit."
    exit 1
fi

mkdir -p ./RESULTS/$EVAL_USER

if [ "$run_mode" = "full" ]; then
    output_file=./RESULTS/$EVAL_USER/fig-10-full.png
else
    output_file=./RESULTS/$EVAL_USER/fig-10.png
fi

echo "Generating figure 10..."

source .venv/bin/activate
python scripts/utils.py $EVAL_LOG_DIR/$EVAL_USER fig10 $output_file > $output_file.log 

echo "Figure 10 generated and saved to $output_file"

##############################################

