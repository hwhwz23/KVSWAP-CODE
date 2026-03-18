#!/bin/bash
set -e

# check is pwd is quality
if [ "$(basename "$(pwd)")" != "quality" ]; then
    echo "Error: Please run this script from the quality directory."
    exit 1
fi

echo "Creating virtual environment..."

if ! command -v uv &> /dev/null
then
    echo "Error: uv is not installed. Please install uv before running this script."
    exit 1
fi

if ! command -v git-lfs &> /dev/null
then
    echo "Error: git-lfs is not installed. Please install git-lfs before running this script."
    exit 1
fi


############################################################################
echo "Installing dependencies..."
uv venv --python 3.10
source .venv/bin/activate
uv pip install pip
uv pip install torch==2.6.0
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install -r requirements.txt
pushd src/shadowkv
python setup.py build_ext --inplace
popd
echo "--------------------------------"


############################################################################
echo "Downloading benchmark datasets and adapter weights..." 
git lfs pull
echo "--------------------------------"


############################################################################
echo "Downloading models..."
mkdir -p MODELS

git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct MODELS/Llama-3.1-8B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-4B MODELS/Qwen3-4B
git clone https://huggingface.co/Qwen/Qwen3-8B MODELS/Qwen3-8B
git clone https://huggingface.co/Qwen/Qwen3-14B MODELS/Qwen3-14B

git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct MODELS/Qwen2.5-VL-3B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct MODELS/Qwen2.5-VL-7B-Instruct
git clone https://huggingface.co/OpenGVLab/InternVL3-14B MODELS/InternVL3-14B

echo "--------------------------------"


############################################################################
echo "Initialization completed!"

