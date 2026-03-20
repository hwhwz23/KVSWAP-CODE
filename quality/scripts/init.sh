#!/bin/bash
set -e

# check is pwd is quality
if [ "$(basename "$(pwd)")" != "quality" ]; then
    echo "Error: Please run this script from the quality directory."
    exit 1
fi


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

if [ ! -d .venv ]; then
    uv venv --python 3.10
fi
source .venv/bin/activate
uv pip install pip setuptools
uv pip install torch==2.6.0
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install -r requirements.txt
pushd src/shadowkv
python setup.py build_ext --inplace
popd
echo "--------------------------------"


############################################################################
echo "Downloading adapter weights..." 
git lfs pull --include="adapters/**/*.pt"
echo "--------------------------------"
############################################################################


echo "Initialization completed!"

