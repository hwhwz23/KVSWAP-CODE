#!/bin/bash
set -e


if [ "$(basename "$(pwd)")" != "engine" ]; then
    echo "Error: Please run this script from the engine directory."
    exit 1
fi


# if ! command -v uv &> /dev/null
# then
#     echo "Error: uv is not installed. Please install uv before running this script."
#     exit 1
# fi

##########################################################
echo "Installing dependencies..."

if [ ! -d .venv ]; then
    uv venv --python 3.10
fi

# source .venv/bin/activate


# uv pip install notebook jupyterlab
# jupyter notebook --generate-config
# python -c "from jupyter_server.auth import passwd; print(passwd())"
# TODO...


nohup jupyter lab --no-browser --ip=127.0.0.1 --port=8888 > jupyter.log 2>&1 &

# TODO...


echo "Building shadowkv..."
pushd src/shadowkv
python setup.py build_ext --inplace
popd

echo "Building Liburing..."
pushd src/Liburing
pip install -e .
popd

echo "--------------------------------"

exit

# for zram in $(ls /dev/zram*); do
#     echo $zram
#     sudo swapoff $zram
#     sudo zramctl --reset $zram
# done

echo 1 | sudo tee /sys/module/nvme_core/parameters/io_timeout


if [ -z "$EMMC_OFFLOAD_DIR" ]; then
    echo "Error: EMMC_OFFLOAD_DIR is not set"
    exit 1
fi

if [ -z "$EMMC_DEV_NAME" ]; then
    echo "Error: EMMC_DEV_NAME is not set"
    exit 1
fi

if [ -z "$NVME_OFFLOAD_DIR" ]; then
    echo "Error: NVME_OFFLOAD_DIR is not set"
    exit 1
fi  

if [ -z "$NVME_DEV_NAME" ]; then
    echo "Error: NVME_DEV_NAME is not set"
    exit 1
fi

echo "Mounting eMMC..."
mkdir -p $EMMC_OFFLOAD_DIR
sudo mount /dev/$EMMC_DEV_NAME $EMMC_OFFLOAD_DIR

echo "Mounting NVMe..."
mkdir -p $NVME_OFFLOAD_DIR
sudo mount /dev/$NVME_DEV_NAME $NVME_OFFLOAD_DIR


##########################################################

if [ -z "$MODEL_PATH_BASE_HF" ]; then
    echo "Error: MODEL_PATH_BASE_HF is not set"
    exit 1
fi

if [ -z "$MODEL_PATH_BASE" ]; then
    echo "Error: MODEL_PATH_BASE is not set"
    exit 1
fi

##########################################################
echo "Making np weights..."
MODEL_LIST=(
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
    "Qwen3-14B"
)

for model in "${MODEL_LIST[@]}"; do
    python scripts/make_np_weights.py --hf_model_path $MODEL_PATH_BASE_HF/$model --save_dir $MODEL_PATH_BASE
done

echo "Done: Making np weights"

##########################################################

echo "Check disk free space..."
# TODO 




