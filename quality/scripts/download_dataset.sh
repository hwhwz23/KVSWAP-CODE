#!/usr/bin/env bash

set -e

mode="${1:-quick}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MLVU_SUBSET_FILE="${SCRIPT_DIR}/mlvu_subset.txt"
MLVU_BASE_GLOB="quality/bench/MLVU/MVLU_DATA/MLVU/video"

if [ "${mode}" == "full" ]; then
    echo "This will download all datasets (~130 GB), it may take a while..."
    echo "--------------------------------"
    echo "Downloading RULER dataset..."
    git -C "${GIT_ROOT}" lfs pull --include="quality/bench/RULER/DATA/Llama/synthetic/32768/**/*.jsonl"
    echo "--------------------------------"
    echo "Downloading MLVU dataset..."
    git -C "${GIT_ROOT}" lfs pull --include="${MLVU_BASE_GLOB}/**/*.mp4"
    echo "--------------------------------"
else
    echo "This will download a subset of datasets (~25 GB), it may take a while..."
    echo "--------------------------------"
    echo "Downloading RULER dataset..."
    git -C "${GIT_ROOT}" lfs pull --include="quality/bench/RULER/DATA/Llama/synthetic/32768/**/*.jsonl"
    echo "--------------------------------"
    echo "Downloading a subset of MLVU dataset..."
    if [ ! -f "${MLVU_SUBSET_FILE}" ]; then
        echo "Error: MLVU subset list not found: ${MLVU_SUBSET_FILE}" >&2
        exit 1
    fi
    mlvu_include=""
    while IFS= read -r line || [ -n "${line}" ]; do
        # trim, strip optional quotes, skip blanks and comments
        line="$(echo "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')"
        [ -z "${line}" ] && continue
        [[ "${line}" == \#* ]] && continue
        pat="${MLVU_BASE_GLOB}/**/${line}"
        if [ -n "${mlvu_include}" ]; then
            mlvu_include="${mlvu_include},${pat}"
        else
            mlvu_include="${pat}"
        fi
    done < "${MLVU_SUBSET_FILE}"
    if [ -z "${mlvu_include}" ]; then
        echo "Error: no filenames in ${MLVU_SUBSET_FILE}" >&2
        exit 1
    fi
    # echo "mlvu_include: ${mlvu_include}"
    git -C "${GIT_ROOT}" lfs pull --include="${mlvu_include}"
    echo "--------------------------------"
fi

echo "Dataset download completed!"
echo "--------------------------------"

