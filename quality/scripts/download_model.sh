#!/usr/bin/env bash
set -euo pipefail

echo "Downloading models..."

MODELS_DIR="${MODELS_DIR:-MODELS}"
mkdir -p "${MODELS_DIR}"

mode="${1:-}"

# If INTERACTIVE_PROMPT=1 and the script is run in a terminal,
# we will ask whether you already have model weights locally and symlink them.
INTERACTIVE_PROMPT="${INTERACTIVE_PROMPT:-0}"

has_model_dir() {
  local dir="$1"
  if [ ! -d "${dir}" ]; then
    return 1
  fi
  # Heuristic: model folder should contain at least one common metadata file.
  if [ -f "${dir}/config.json" ] || [ -f "${dir}/generation_config.json" ] || [ -d "${dir}/model" ]; then
    return 0
  fi
  return 1
}

ensure_model() {
  local model_name="$1"
  local dest_dir="$2"
  local hf_repo="$3"
  local full_dest="${MODELS_DIR}/${dest_dir}"

  if has_model_dir "${full_dest}"; then
    echo "Already present: ${model_name} -> ${full_dest}"
    return 0
  fi

  echo "Missing: ${model_name} -> ${full_dest}"

  if [ "${INTERACTIVE_PROMPT}" == "1" ] && [ -t 0 ]; then
    echo "Do you already have ${model_name} weights locally?"
    echo "Type 'y' to symlink from a local path; otherwise press Enter to download."
    read -r ans || true
    ans="$(echo "${ans}" | tr '[:upper:]' '[:lower:]')"
    if [ "${ans}" == "y" ] || [ "${ans}" == "yes" ]; then
      echo "Enter local path to the ${model_name} folder:"
      read -r local_path || true
      if [ -n "${local_path}" ] && [ -d "${local_path}" ]; then
        ln -sfn "${local_path}" "${full_dest}"
        echo "Symlink created: ${full_dest} -> ${local_path}"
        return 0
      else
        echo "Invalid local path. Will download instead."
      fi
    fi
  fi

  echo "Cloning from: ${hf_repo}"
  # Requires git-lfs for LFS-managed files.
  GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 "${hf_repo}" "${full_dest}"
}

download_model() {
  ensure_model "Llama-3.1-8B-Instruct"      "Llama-3.1-8B-Instruct"      "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
  ensure_model "Qwen3-4B"                  "Qwen3-4B"                  "https://huggingface.co/Qwen/Qwen3-4B"
  ensure_model "Qwen3-8B"                  "Qwen3-8B"                  "https://huggingface.co/Qwen/Qwen3-8B"
  ensure_model "Qwen2.5-VL-3B-Instruct"   "Qwen2.5-VL-3B-Instruct"   "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"
  ensure_model "Qwen2.5-VL-7B-Instruct"   "Qwen2.5-VL-7B-Instruct"   "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
}

download_model2() {
  ensure_model "Qwen3-14B"                "Qwen3-14B"                "https://huggingface.co/Qwen/Qwen3-14B"
  ensure_model "InternVL3-14B"           "InternVL3-14B"           "https://huggingface.co/OpenGVLab/InternVL3-14B"
}

if [ "${mode}" == "full" ]; then
  download_model
  download_model2
else
  download_model
fi

echo "--------------------------------"