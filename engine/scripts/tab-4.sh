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

################FlexGen################
run_flexgen(){
    return 0
    echo "Running FlexGen..."
    RUN_ARGS=L1 
    USE_TOKEN_CACHE=0
    START_LAYER=none
    LR_PROJ_MODE=none
    TOKEN_GROUP=1
    MAX_NUM_KV=0
    REUSE_BUDGET=0
    RATIO=none
    ./scripts/eval.sh $TEST_MODEL $DISK_TYPE $LR_PROJ_MODE $TOTAL_LEN $TOKEN_GROUP \
        $MAX_NUM_KV $SEED $REUSE_BUDGET $RATIO $START_LAYER $USE_TOKEN_CACHE \
        $RUN_ARGS "$BATCH_LIST"
    echo "FlexGen done."
}

################Infinigen/Infinigen*(+reuse)################
run_infinigen(){
    echo "Running Infinigen..."
    RUN_ARGS=L4 
    USE_TOKEN_CACHE=0
    START_LAYER=0-curr-emb
    LR_PROJ_MODE=base
    TOKEN_GROUP=1
    MAX_NUM_KV=$MAX_KV_FETCH
    SKEW_RARIO=$INFI_RATIO
    REUSE_BUDGET=0
    ./scripts/eval.sh $TEST_MODEL $DISK_TYPE $LR_PROJ_MODE $TOTAL_LEN $TOKEN_GROUP \
        $MAX_NUM_KV $SEED $REUSE_BUDGET $SKEW_RARIO $START_LAYER $USE_TOKEN_CACHE \
        $RUN_ARGS "$BATCH_LIST"

    REUSE_BUDGET=$MAX_KV_FETCH 
    ./scripts/eval.sh $TEST_MODEL $DISK_TYPE $LR_PROJ_MODE $TOTAL_LEN $TOKEN_GROUP \
        $MAX_NUM_KV $SEED $REUSE_BUDGET $SKEW_RARIO $START_LAYER $USE_TOKEN_CACHE \
        $RUN_ARGS "$BATCH_LIST"
    echo "Infinigen done."
}

################ShadowKV#######################
run_shadowkv(){
    echo "Running ShadowKV..."
    # chunk_size=8
    # rank=160
    chunk_size=16
    rank=40
    ./src/shadowkv/run_shadowkv.sh $TEST_MODEL $DISK_TYPE $TOTAL_LEN \
        $SEED $MAX_NUM_KV $chunk_size $rank "$BATCH_LIST"
    echo "ShadowKV done."
}

################KVSwap#######################
run_kvswap(){
    echo "Running KVSwap..."
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
    echo "KVSwap done."
}




#############################################
TEST_MODEL=Llama-3.1-8B-Instruct
MAX_KV_FETCH=400
KVSWAP_RATIO=1
INFI_RATIO=0.125

SEQ_LIST=(16384 32768)

clear_offload_dir nvme
clear_offload_dir emmc
COUNT=0
while IFS= read -r seed; do
    COUNT=$((COUNT + 1))
    for TOTAL_LEN in "${SEQ_LIST[@]}"; do
        echo "Running with sequence length: $TOTAL_LEN and seed: $seed"
        SEED=$seed
        #########################################################
        export MAX_ALLOC_KV_SIZE=$((1024*1024*2048))
        BATCH_LIST="1 2 4 8 16"
        DISK_TYPE=nvme
        KVSWAP_TG=4
        run_flexgen
        run_infinigen
        run_kvswap
        #########################################################
        export MAX_ALLOC_KV_SIZE=$((1024*1024*768))
        BATCH_LIST="1 2 4"
        DISK_TYPE=emmc
        KVSWAP_TG=8
        run_flexgen
        run_infinigen
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
        export MAX_ALLOC_KV_SIZE=$((1024*1024*2048))
        BATCH_LIST="1 2 4 8 16"
        DISK_TYPE=nvme
        run_shadowkv
        #########################################################
        export MAX_ALLOC_KV_SIZE=$((1024*1024*768))
        BATCH_LIST="1 2 4 8"
        DISK_TYPE=emmc
        run_shadowkv
        #########################################################
    done
    if [ "$COUNT" -ge "$MAX_COUNT" ]; then
        break
    fi
done < "./data/seeds.txt"

#############################################
# run vLLM
SEQ_LIST="16384,32768"
BATCH_LIST="1,2,4,8,16"

echo "Running vLLM..."
./scripts/run_vllm.sh $TEST_MODEL $DISK_TYPE $SEQ_LIST $BATCH_LIST
echo "vLLM done."

#############################################
# Output Results




##############################################