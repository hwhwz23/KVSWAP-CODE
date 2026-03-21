#!/usr/bin/env bash

set -e

mode="${1:-quick}"


if [ "${mode}" == "full" ]; then
    echo "This will download all datasets, it may take a while..."
    echo "--------------------------------"
    echo "Downloading RULER dataset..."
    git lfs pull --include="bench/RULER/DATA/Llama/synthetic/32768/**/*.jsonl"
    echo "--------------------------------"
    echo "Downloading MLVU dataset..."
    git lfs pull --include="bench/bench/MLVU/MVLU_DATA/MLVU/video/**/*.mp4"
    echo "--------------------------------"
else
    echo "This will download a subset of datasets, it may take a while..."
    echo "--------------------------------"
    echo "Downloading RULER dataset..."
    git lfs pull --include="bench/RULER/DATA/Llama/synthetic/32768/**/*.jsonl"
    echo "--------------------------------"
    echo "Downloading a subset of MLVU dataset..."
    # TODO only download video files with names of xxxxx

    echo "--------------------------------"
fi

echo "Dataset download completed!"
echo "--------------------------------"

