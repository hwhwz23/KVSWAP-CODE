
PASSWD=
TEST_INPUT_PATH=./data/test_inputs

HOSTNAME=$(hostname)
echo "Running on host: $HOSTNAME"

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

run(){
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

    echo $PASSWD | sudo -S sh -c "blockdev --setra $READAHEAD /dev/${DISK_DEV_NAME_}"
    echo "/dev/${DISK_DEV_NAME_} READAHEAD is: "
    sudo -S sh -c "blockdev --getra /dev/${DISK_DEV_NAME_}"

    model_name=$(basename ${model_path})
    if [ "$offload_device" = "disk" ]; then
        log_file=$log_dir/${disk_type_}/${model_name}/budget${budget}/seed${seed}/${min_prompt_len}_bsz${bsz}_gen${genlen}_chunk${chunk_size}_r${rank}.log
    else
        echo "Should use disk offload. Exit."
        exit 1
    fi
    mkdir -p $(dirname $log_file)
    if grep -q "Throughput:" "$log_file"; then
        echo "Log file $log_file already contains throughput results. Skipping this run."
        return
    fi
    cache_dir=${OFFLOAD_DIR_}
    python src/shadowkv/test/e2e_jetson.py --model_path ${model_path} --min_prompt_len ${min_prompt_len} \
        --bsz ${bsz} --budget ${budget} --genlen ${genlen} --input_path ${TEST_INPUT_PATH} \
        --chunk_size ${chunk_size} --rank ${rank} --cache_dir ${cache_dir} \
        --offload_device ${offload_device} --log_file ${log_file} --seed ${seed}
}


####################################################################
set_powermode
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES="0"
source .venv/bin/activate
offload_device="disk"

DEV=$HOSTNAME
log_dir=./exps/logs/shadowkv/${DEV}
mkdir -p $log_dir

READAHEAD=0
echo 1 | sudo tee /sys/module/nvme_core/parameters/io_timeout

if [ -z "$MODEL_PATH_BASE_HF" ]; then
  echo "MODEL_PATH_BASE_HF is not set. Exit."
  exit 1
fi

TEST_MODEL=$1
disk_type_=$2
TOTAL_LEN=$3
seed=$4
budget=$5
chunk_size=$6
rank=$7
IFS=' ' read -r -a BATCHSIZE_LIST <<< "$8"

####################################################################
echo TEST_MODEL=$TEST_MODEL
echo disk_type_=$disk_type_
echo TOTAL_LEN=$TOTAL_LEN
echo seed=$seed
echo budget=$budget
echo chunk_size=$chunk_size
echo rank=$rank
echo BATCHSIZE_LIST=${BATCHSIZE_LIST[@]}

####################################################################
model_path=${MODEL_PATH_BASE_HF}/${TEST_MODEL}
genlen=100
min_prompt_len=$((TOTAL_LEN - genlen))


SKIP_MODEL_RUN=${SKIP_MODEL_RUN:-0}
if [ "$SKIP_MODEL_RUN" = "1" ]; then
  echo "SKIP_MODEL_RUN=1, skipping model run"
  exit 0
fi

for bsz in ${BATCHSIZE_LIST[@]}; do
  echo Evaluating bsz=$bsz
  run
  sleep 3
done


echo "Evaluation finished."


