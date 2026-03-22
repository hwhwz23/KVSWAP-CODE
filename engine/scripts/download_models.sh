#!/usr/bin/env bash
set -e

if [ -z "${MODEL_PATH_BASE_HF}" ]; then
  echo "MODEL_PATH_BASE_HF is not set"
  exit 1
fi

# If INTERACTIVE_PROMPT=1 and the script is run in a terminal,
# we will ask whether you already have model weights locally and symlink them.
INTERACTIVE_PROMPT="${INTERACTIVE_PROMPT:-1}"

has_weight_files() {
  local dir="$1"
  shopt -s nullglob
  # Common Hugging Face weight file patterns (supports sharded weights).
  local candidates=(
    "${dir}/model*.safetensors"
    "${dir}/pytorch_model*.bin"
    "${dir}/*.bin"
    "${dir}/model*.pt"
    "${dir}/*.pt"
    "${dir}/model*.pth"
    "${dir}/*.pth"
    "${dir}/model*.safetensors.index.json"
    "${dir}/pytorch_model*.bin.index.json"
  )
  # Each pattern may expand to 0, 1, or many files; [ -e $pattern ] breaks on multi-match.
  for c in "${candidates[@]}"; do
    for f in $c; do
      if [ -e "$f" ]; then
        return 0
      fi
    done
  done
  return 1
}

is_valid_model_dir() {
  local dir="$1"
  if [ ! -d "${dir}" ]; then
    return 1
  fi
  # Require config.json AND weight files.
  if [ ! -f "${dir}/config.json" ]; then
    return 1
  fi
  has_weight_files "${dir}"
}

validate_local_path() {
  local model_name="$1"
  local dir="$2"

  if [ ! -d "$dir" ]; then
    echo "Invalid path: $dir (not a directory)"
    return 1
  fi

  # Must look like a HF model folder.
  if [ ! -f "${dir}/config.json" ]; then
    echo "Invalid local folder for ${model_name}: missing config.json"
    return 1
  fi

  if ! has_weight_files "$dir"; then
    echo "Invalid local folder for ${model_name}: no weight files found (expected *.safetensors or *.bin, etc.)"
    return 1
  fi

  return 0
}

ensure_model() {
  local model_name="$1"
  local dest_dir="$2"
  local hf_repo="$3"
  local full_dest="${MODEL_PATH_BASE_HF}/${dest_dir}"

  if is_valid_model_dir "${full_dest}"; then
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
      if [ -n "${local_path}" ]; then
        if validate_local_path "${model_name}" "${local_path}"; then
          ln -sfn "${local_path}" "${full_dest}"
          echo "Symlink created: ${full_dest} -> ${local_path}"
          return 0
        else
          echo "Local path does not look like a valid model. Will download instead."
        fi
      else
        echo "Empty local path. Will download instead."
      fi
    fi
  fi

  echo "Cloning from: ${hf_repo}"
  # Requires git-lfs for LFS-managed files.
  GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 "${hf_repo}" "${full_dest}"
}


echo "Preparing model weights..."
echo "--------------------------------"

ensure_model "Llama-3.1-8B-Instruct"      "Llama-3.1-8B-Instruct"      "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
ensure_model "Llama-3.2-3B-Instruct"      "Llama-3.2-3B-Instruct"      "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
ensure_model "Qwen3-14B"                "Qwen3-14B"                "https://huggingface.co/Qwen/Qwen3-14B"

echo "Done preparing model weights"
echo "--------------------------------"
