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

if [ -z "$EVAL_USER" ]; then
  echo "EVAL_USER is not set. This is set for storing results. Exit."
  exit 1
fi

if [ -z "$EVAL_LOG_DIR" ]; then
  echo "EVAL_LOG_DIR is not set. This is set for storing results. Exit."
  exit 1
fi

OUTPUT_PATH=$EVAL_LOG_DIR/$EVAL_USER/vllm_results
mkdir -p $OUTPUT_PATH

LOG_OUT=$OUTPUT_PATH/$TEST_MODEL.log

SKIP_MODEL_RUN=${SKIP_MODEL_RUN:-0}
if [ "$SKIP_MODEL_RUN" = "1" ]; then
  echo "SKIP_MODEL_RUN=1, skipping model run"
  exit 0
fi


echo "Running vLLM with model: $TEST_MODEL"

python src/run_vllm.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH \
    --seqlen-list $SEQLEN_LIST --batch-list $BATCH_LIST > $LOG_OUT 2>&1

sleep 3

echo "Done"

