#!/bin/bash
set -e


HOSTNAME=$(hostname)
echo "Running on host: $HOSTNAME"

TEST_INPUT_PATH=./data/test_inputs

if [ -z "$EVAL_USER" ]; then
  echo "EVAL_USER is not set. This is set for storing results. Exit."
  exit 1
fi

if [ -z "$EVAL_LOG_DIR" ]; then
  echo "EVAL_LOG_DIR is not set. This is set for storing results. Exit."
  exit 1
fi

SET_LOG_DIR=$EVAL_LOG_DIR/$EVAL_USER/logs
mkdir -p $SET_LOG_DIR

PASSWD=

#################################################################################
####################UTILS#############################################################

set_io() {
  IFS=',' read -ra disk_types <<< "$DISK_TYPE"
  OFFLOAD_DIR=''
  DISK_DEV_NAME=''
  OFFLOAD_DIR_LIST=()
  DISK_DEV_NAME_LIST=()
  for disk_type_ in "${disk_types[@]}"; do
    if [ "$disk_type_" = "emmc" ]; then
      if [ -z "$EMMC_OFFLOAD_DIR" ]; then
        echo "EMMC_OFFLOAD_DIR is not set. Exit."
        exit 1
      fi
      OFFLOAD_DIR_=$EMMC_OFFLOAD_DIR
      if [ -z "$EMMC_DEV_NAME" ]; then
        echo "EMMC_DEV_NAME is not set. Exit."
        exit 1
      fi
      DISK_DEV_NAME_=$EMMC_DEV_NAME
    elif [ "$disk_type_" = "usb" ]; then
      if [ -z "$USB_OFFLOAD_DIR" ]; then
        echo "USB_OFFLOAD_DIR is not set. Exit."
        exit 1
      fi
      OFFLOAD_DIR_=$USB_OFFLOAD_DIR
      if [ -z "$USB_DEV_NAME" ]; then
        echo "USB_DEV_NAME is not set. Exit."
        exit 1
      fi
      DISK_DEV_NAME_=$USB_DEV_NAME
    elif [ "$disk_type_" = "nvme" ]; then
      if [ -z "$NVME_OFFLOAD_DIR" ]; then
        echo "NVME_OFFLOAD_DIR is not set. Exit."
        exit 1
      fi
      OFFLOAD_DIR_=$NVME_OFFLOAD_DIR
      if [ -z "$NVME_DEV_NAME" ]; then
        echo "NVME_DEV_NAME is not set. Exit."
        exit 1
      fi
      DISK_DEV_NAME_=$NVME_DEV_NAME
    else
      echo "Invalid disk type $disk_type_. Exit."
      exit 1
    fi
    mkdir -p $OFFLOAD_DIR_
    OFFLOAD_DIR=${OFFLOAD_DIR},${OFFLOAD_DIR_}
    DISK_DEV_NAME=${DISK_DEV_NAME},${DISK_DEV_NAME_}
    OFFLOAD_DIR_LIST+=("$OFFLOAD_DIR_")
    DISK_DEV_NAME_LIST+=("$DISK_DEV_NAME_")
  done
  OFFLOAD_DIR="${OFFLOAD_DIR#,}"
  DISK_DEV_NAME="${DISK_DEV_NAME#,}"
}


set_powermode() { 
  # For Jetson Orin AGX experiments we require MAXN mode.
  if ! command -v nvpmodel >/dev/null 2>&1; then
    echo "nvpmodel not found; cannot verify power mode"
    exit 1
  fi

  local current=""
  current="$(nvpmodel -q --verbose 2>/dev/null | awk -F'NV Power Mode: ' '/NV Power Mode:/{print $2; exit}' | tr -d '\r' | xargs)"

  if [[ "$current" != "MAXN" ]]; then
    echo "Power mode is '$current', expected 'MAXN'. Exit."
    exit 1
  fi

  echo "Power mode verified: MAXN"
  return 0
}


set_base_dir() {
  DIR_NAME=${DEV}/${DISK_TYPE}/${MODEL_NAME}
  LOG_DIR_=$SET_LOG_DIR/log/${DIR_NAME}
  mkdir -p $LOG_DIR_
  if [ "$NV_PROFILE" = 1 ]; then
    NVVP_DIR_=$SET_LOG_DIR/nvvp/${DIR_NAME}
    NVVP_LOG_DIR_=$SET_LOG_DIR/nvvp_log/${DIR_NAME}
    mkdir -p $NVVP_DIR_
    mkdir -p $NVVP_LOG_DIR_
  fi
}

set_dir() {
  DIR_NAME=${LR_PROJ_MODE}/tg${TOKEN_GROUP}-ru${REUSE_BUDGET}/seed${SEED}/
  LOG_DIR=${LOG_DIR_}/${DIR_NAME}
  echo "****LOG_DIR="$LOG_DIR
  mkdir -p $LOG_DIR
  if [ "$NV_PROFILE" = 1 ]; then
    NVVP_DIR=${NVVP_DIR_}/${DIR_NAME}
    NVVP_LOG_DIR=${NVVP_LOG_DIR_}/${DIR_NAME}
    mkdir -p $NVVP_DIR
    mkdir -p $NVVP_LOG_DIR
  fi
}

cleanup() {
    echo "Cleaning up..."
    if [[ -n "$PROGRAM_PID" ]]; then
        echo "Killing program (PID: $PROGRAM_PID)"
        kill -9 "$PROGRAM_PID" 2>/dev/null
    fi
    exit 0
}

#################################################################################
#################################################################################

run_once() {
    set_dir
    if [ "$NV_PROFILE" = 1 ]; then
      echo "Set GEN_LEN to 50 for NVVP profiling"
      GEN_LEN=50
    fi
    CMD="--percent 100 0 0 0 100 0 --model_path $MODEL_PATH --offload_dir $OFFLOAD_DIR --prompt_len $PROMPT_LEN \
          --gen_len $GEN_LEN --gpu_batch_size $BATCHSIZE --num_gpu_batches 1 --test_input_path $TEST_INPUT_PATH --run_args $RUN_ARGS \
          --lr_proj_mode $LR_PROJ_MODE \
          --use_token_cache $USE_TOKEN_CACHE --dk_wr $DK_WR --dk_rd $DK_RD \
          --reuse_budget $REUSE_BUDGET --token_group $TOKEN_GROUP \
          --disk_dev_name $DISK_DEV_NAME --batch_split $BATCH_SPLIT  --seed $SEED "

    RUN_INFO=${BATCHSIZE}-${PROMPT_LEN}-${GEN_LEN}_${RUN_ARGS}_${DK_WR}_${DK_RD}_${BATCH_SPLIT}
    if [ "$LR_PROJ_MODE" = "none" ]; then
      RUN_INFO=${RUN_INFO}_full
    else
      RUN_INFO=${RUN_INFO}_${MAX_NUM_KV}_${START_LAYER}
      CMD=$CMD" --max_num_kv $MAX_NUM_KV --start_layer $START_LAYER "

      if [ "$LR_PROJ_MODE" = "base" ]; then
        SKEW_MATRIX_PATH=${MODEL_PATH_BASE}/infinigen_skew/skewing_matrix_c4_20/${MODEL_NAME}.pt
        SKEW_PARTIAL_IDX_PATH=${MODEL_PATH_BASE}/infinigen_skew/skewing_idx_c4_20/${MODEL_NAME}_${SKEW_RARIO}
        RUN_INFO=${RUN_INFO}_p${SKEW_RARIO}
        CMD=$CMD" --skew_matrix_path $SKEW_MATRIX_PATH --skew_partial_idx_path $SKEW_PARTIAL_IDX_PATH "
      else
        LR_MODE="${LR_PROJ_MODE##*_}"
        LR_PROJ_PATH=${MODEL_PATH_BASE}/lowrank_proj_post_rope_c4_20/${MODEL_NAME}_${LR_MODE}_${LR_PROJ_RATIO}
        RUN_INFO=${RUN_INFO}_p${LR_PROJ_RATIO}
        CMD=$CMD" --lr_proj_path $LR_PROJ_PATH "
      fi
    fi

    if [ "$USE_TOKEN_CACHE" = 1 ]; then
      RUN_INFO=${RUN_INFO}_tokce
    fi

    CMD=$CMD" --run_info $LOG_DIR/$RUN_INFO"
    echo $RUN_INFO

    if [ "$NV_PROFILE" = 1 ]
    then
      LOG_OUT=$NVVP_LOG_DIR/$RUN_INFO".log"
      if grep -q "Throughput Total:" "$LOG_OUT"; then
        echo "Log file $LOG_OUT already contains throughput results. Skipping this run."
        return
      fi
    else
      LOG_OUT=$LOG_DIR/$RUN_INFO".log"
      if grep -q "Throughput Total:" "$LOG_OUT"; then
        echo "Log file $LOG_OUT already contains throughput results. Skipping this run."
        return
      fi
    fi

    for DISK_DEV_NAME_ in "${DISK_DEV_NAME_LIST[@]}"; do
      echo $PASSWD | sudo -S sh -c "blockdev --setra $READAHEAD /dev/${DISK_DEV_NAME_}"
      echo "/dev/${DISK_DEV_NAME_} READAHEAD is: "
      sudo -S sh -c "blockdev --getra /dev/${DISK_DEV_NAME_}"
    done

    if [ "$NV_PROFILE" = 1 ]
    then
      echo "CMD="$CMD > $LOG_OUT
      $NVPROF_CMD$NVVP_DIR/$RUN_INFO $PYTHON_EXE main.py $CMD --nv_profile 1 >> $LOG_OUT &
    else
      echo "CMD="$CMD > $LOG_OUT
      $PYTHON_EXE main.py $CMD --nv_profile 0 >> $LOG_OUT &
    fi
    PROGRAM_PID=$!
    wait $PROGRAM_PID
}


PYTHON_EXE=python
NVPROF_TRACE=cuda,nvtx
NVPROF_CMD="nsys profile -w true -t $NVPROF_TRACE -s none --export=none -b none --cpuctxsw none
                        --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=none -x true -f true -o "

export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES="0"

source .venv/bin/activate

#####################################################
DEV=${HOSTNAME}
BATCH_SPLIT=1
NV_PROFILE=0
READAHEAD=0
DK_RD=clear
DK_WR=none
# DK_RD=none
set_io


set_powermode
trap cleanup SIGINT

#####################################################
TEST_MODEL=$1
DISK_TYPE=$2
LR_PROJ_MODE=$3
TOTAL_LEN=$4
TOKEN_GROUP=$5
MAX_NUM_KV=$6
SEED=$7
REUSE_BUDGET=$8
RATIO=$9
START_LAYER=$10
USE_TOKEN_CACHE=$11
# L0: original; L1: map_dict; L2: +copy_key(finer_sync); L3: +ahead_prefetch; L4: +finer_prefetch_sync;
RUN_ARGS=$12
IFS=' ' read -r -a BATCHSIZE_LIST <<< "$13"
##################################################
echo TEST_MODEL=$TEST_MODEL
echo DISK_TYPE=$DISK_TYPE
echo LR_PROJ_MODE=$LR_PROJ_MODE
echo TOTAL_LEN=$TOTAL_LEN
echo TOKEN_GROUP=$TOKEN_GROUP
echo MAX_NUM_KV=$MAX_NUM_KV
echo SEED=$SEED
echo REUSE_BUDGET=$REUSE_BUDGET
echo RATIO=$RATIO
echo START_LAYER=$START_LAYER
echo USE_TOKEN_CACHE=$USE_TOKEN_CACHE
echo RUN_ARGS=$RUN_ARGS
echo BATCHSIZE_LIST=$BATCHSIZE_LIST
##################################################
if [ "$LR_PROJ_MODE" = "base" ]; then
  SKEW_RARIO=$RATIO
else
  LR_PROJ_RATIO=$RATIO
fi
##################################################
GEN_LEN=100
PROMPT_LEN=$((TOTAL_LEN - GEN_LEN))
# check MODEL_PATH_BASE is set
if [ -z "$MODEL_PATH_BASE" ]; then
  echo "MODEL_PATH_BASE is not set. Exit."
  exit 1
fi

MODEL_PATH=${MODEL_PATH_BASE}/${TEST_MODEL}
MODEL_NAME=$(basename $MODEL_PATH)
set_base_dir

SKIP_MODEL_RUN=${SKIP_MODEL_RUN:-0}
if [ "$SKIP_MODEL_RUN" = "1" ]; then
  echo "SKIP_MODEL_RUN=1, skipping model run"
  exit 0
fi

for BATCHSIZE in ${BATCHSIZE_LIST[@]}; do
  echo Evaluating BATCHSIZE=$BATCHSIZE
  run_once
  sleep 3
done



echo Evaluation finished.
