#!/bin/bash
set -e

export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES="0"

source .venv/bin/activate

TEST_MODEL=$1
SEQLEN_LIST=$2
BATCH_LIST=$3

# check MODEL_PATH_BASE_HF is set
if [ -z "$MODEL_PATH_BASE_HF" ]; then
  echo "MODEL_PATH_BASE_HF is not set. Exit."
  exit 1
fi

MODEL_PATH=${MODEL_PATH_BASE_HF}/${TEST_MODEL}

OUTPUT_PATH=./exps/vllm_results
mkdir -p $OUTPUT_PATH

LOG_OUT=$OUTPUT_PATH/$TEST_MODEL.log

echo "Running vLLM with model: $TEST_MODEL"

python src/run_vllm.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH \
    --seqlen-list $SEQLEN_LIST --batch-list $BATCH_LIST > $LOG_OUT 2>&1

echo "Done"

