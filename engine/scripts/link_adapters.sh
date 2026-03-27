#!/usr/bin/env bash
set -e

#########################SHOULD TEST BEFORE PUBLISHING

if [ -z "${MODEL_PATH_BASE}" ]; then
  echo "MODEL_PATH_BASE is not set"
  exit 1
fi

# first download then symlink
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# git -C "${GIT_ROOT}" lfs pull --include="engine/data/adapters/**/*.pt"

mkdir -p "${MODEL_PATH_BASE}"
for d in "${GIT_ROOT}/engine/data/adapters"/*; do
  if [ -d "${d}" ]; then
    ln -snf "${d}" "${MODEL_PATH_BASE}/"
  fi
done

echo "Adapters linked to ${MODEL_PATH_BASE}"


