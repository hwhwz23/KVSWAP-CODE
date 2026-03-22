#!/bin/bash
set -e

check_powermode() { 
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

check_jetson_clocks() {
  # Simple check (no root required):
  # Consider jetson clocks "enabled" if CPU/GPU frequency constraints
  # are pinned to their maximum supported values.

  local cpu_policy="/sys/devices/system/cpu/cpufreq/policy0"
  local gpu_dev="/sys/devices/platform/17000000.gpu/devfreq_dev"

  # ---- CPU ----
  if [ -d "$cpu_policy" ] && \
     [ -f "$cpu_policy/scaling_min_freq" ] && \
     [ -f "$cpu_policy/scaling_max_freq" ] && \
     [ -f "$cpu_policy/scaling_available_frequencies" ]; then
    local cpu_min cpu_max cpu_avail_max
    cpu_min="$(cat "$cpu_policy/scaling_min_freq" 2>/dev/null || echo "")"
    cpu_max="$(cat "$cpu_policy/scaling_max_freq" 2>/dev/null || echo "")"
    cpu_avail_max="$(
      awk '
        { for (i = 1; i <= NF; i++) { v = $i + 0; if (v > max) max = v } }
        END { if (max == "") max = 0; print max }
      ' "$cpu_policy/scaling_available_frequencies" 2>/dev/null
    )"

    if [ -n "$cpu_avail_max" ] && { [ "$cpu_min" != "$cpu_avail_max" ] || [ "$cpu_max" != "$cpu_avail_max" ]; }; then
      echo "CPU clocks not pinned to max: min=$cpu_min max=$cpu_max expected_max=$cpu_avail_max"
      echo "Please enable jetson clocks and try again."
      exit 1
    fi
  else
    echo "CPU cpufreq policy0 info not found."
    exit 1
  fi

  # ---- GPU ----
  if [ -d "$gpu_dev" ] && \
     [ -f "$gpu_dev/min_freq" ] && \
     [ -f "$gpu_dev/max_freq" ] && \
     [ -f "$gpu_dev/available_frequencies" ]; then
    local gpu_min gpu_max gpu_avail_max
    gpu_min="$(cat "$gpu_dev/min_freq" 2>/dev/null || echo "")"
    gpu_max="$(cat "$gpu_dev/max_freq" 2>/dev/null || echo "")"
    gpu_avail_max="$(
      awk '
        { for (i = 1; i <= NF; i++) { v = $i + 0; if (v > max) max = v } }
        END { if (max == "") max = 0; print max }
      ' "$gpu_dev/available_frequencies" 2>/dev/null
    )"

    if [ -n "$gpu_avail_max" ] && { [ "$gpu_min" != "$gpu_avail_max" ] || [ "$gpu_max" != "$gpu_avail_max" ]; }; then
      echo "GPU clocks not pinned to max: min=$gpu_min max=$gpu_max expected_max=$gpu_avail_max"
      echo "Please enable jetson clocks and try again."
      exit 1
    fi
  else
    echo "GPU devfreq_dev info not found."
    exit 1
  fi

  echo "Jetson clocks verified (CPU/GPU pinned to max)."
  return 0
}

check_powermode
check_jetson_clocks

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


# check if OUTPUT_PATH/$TEST_MODEL_results.csv exists
# if [ -f "$OUTPUT_PATH/${TEST_MODEL}_results.csv" ]; then
#   echo "Results already exist in $OUTPUT_PATH/${TEST_MODEL}_results.csv"
#   echo "Skipping model run"
#   exit 0
# fi

echo "Running vLLM with model: $TEST_MODEL"

echo "Clearing system cache..."
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
# exit if no permission
if [ $? -ne 0 ]; then
  echo "Error: No permission to clear system cache."
  echo "We enforce clearing system cache to avoid possible OOM errors when running vLLM."
  echo "Add '<user> ALL=(ALL) NOPASSWD: /usr/bin/tee' to /etc/sudoers to grant permission."
  exit 1  
fi

echo "System cache cleared."
sleep 2

python src/run_vllm.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH \
    --seqlen-list $SEQLEN_LIST --batch-list $BATCH_LIST > $LOG_OUT 

sleep 3

echo "Done"
