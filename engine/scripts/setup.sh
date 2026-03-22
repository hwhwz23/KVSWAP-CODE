#!/bin/bash
set -e

echo "--------------------------------"
echo "This script is only used for setting up your own Orin AGX device."
echo "Do not use it if you are remotely accessing our device."
echo "--------------------------------"

##########################################################
# Require NVIDIA Jetson AGX Orin (model string from device tree)
if [[ ! -r /proc/device-tree/model ]]; then
    echo "Error: Cannot read /proc/device-tree/model. This script expects a Jetson-class device."
    exit 1
fi
MODEL="$(tr -d '\0' < /proc/device-tree/model)"
if [[ "$MODEL" != *"AGX Orin"* ]]; then
    echo "Error: This setup targets NVIDIA Jetson AGX Orin only."
    echo "Detected hardware: ${MODEL}"
    exit 1
fi
echo "Hardware check OK: ${MODEL}"

###############################################################

if [ "$(basename "$(pwd)")" != "engine" ]; then
    echo "Error: Please run this script from the engine directory."
    exit 1
fi

if ! command -v uv &> /dev/null
then
    echo "Error: uv is not installed. Please install uv before running this script."
    exit 1
fi

##########################################################
echo "Installing dependencies..."

if [ ! -d .venv ]; then
    uv venv --python 3.10
    source .venv/bin/activate
    uv pip install pip setuptools
    uv pip install -r requirements.txt
    # check if the wheels are already downloaded
    if [ ! -f ./wheel_pkgs/torch-2.7.0-cp310-cp310-linux_aarch64.whl ]; then
        echo "Error: torch-2.7.0-cp310-cp310-linux_aarch64.whl not found"
        echo "Please refer to wheel_pkgs/readme.txt"
        exit 1
    fi
    if [ ! -f ./wheel_pkgs/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl ]; then
        echo "Error: flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl not found"
        echo "Please refer to wheel_pkgs/readme.txt"
        exit 1
    fi
    if [ ! -f ./wheel_pkgs/triton-3.2.0-cp310-cp310-linux_aarch64.whl ]; then
        echo "Error: triton-3.2.0-cp310-cp310-linux_aarch64.whl not found"
        echo "Please refer to wheel_pkgs/readme.txt"
        exit 1
    fi
    uv pip install ./wheel_pkgs/torch-2.7.0-cp310-cp310-linux_aarch64.whl
    uv pip install ./wheel_pkgs/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl --no-build-isolation
    uv pip install ./wheel_pkgs/triton-3.2.0-cp310-cp310-linux_aarch64.whl --no-deps
    uv pip install notebook jupyterlab
    # transformers==4.51.0
    # tokenizers==0.21

    echo "Building shadowkv..."
    pushd src/shadowkv
    python setup.py build_ext --inplace
    popd

    echo "Building Liburing..."
    pushd src/Liburing
    pip install -e .
    popd

    ###########TODO 
    # uv pip install vllm4kvswap


    echo "Done: Installing dependencies"
else
    echo "Dependencies already installed"
fi

source .venv/bin/activate
echo "--------------------------------"

#############################################################
echo "Starting to setup disk..."

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

# Mount offload devices unless: (1) / is already on that device, or (2) already mounted.
mount_dev_if_needed() {
    local _label="$1"
    local _dev_name="$2"
    local _mnt_dir="$3"
    local _dev_path="/dev/${_dev_name}"

    if [[ ! -b "${_dev_path}" ]]; then
        echo "Error: ${_label} block device ${_dev_path} not found."
        exit 1
    fi

    local _target_real
    _target_real=$(readlink -f "${_dev_path}")

    local _root_src
    _root_real=""
    _root_src=$(findmnt -n -o SOURCE / 2>/dev/null | head -1)
    if [[ -n "${_root_src}" ]]; then
        if [[ "${_root_src}" == "/dev/root" ]]; then
            _root_real=$(readlink -f /dev/root)
        elif [[ -b "${_root_src}" ]]; then
            _root_real=$(readlink -f "${_root_src}")
        elif [[ "${_root_src}" == UUID=* ]]; then
            _root_real=$(blkid -U "${_root_src#UUID=}" 2>/dev/null || true)
            [[ -n "${_root_real}" ]] && _root_real=$(readlink -f "${_root_real}")
        fi
        if [[ -n "${_root_real}" && "${_root_real}" == "${_target_real}" ]]; then
            echo "Skip ${_label}: root filesystem is already on ${_dev_path}"
            return 0
        fi
    fi

    if findmnt -n "${_mnt_dir}" &>/dev/null; then
        echo "Skip ${_label}: ${_mnt_dir} is already a mount point"
        return 0
    fi

    if findmnt -S "${_target_real}" &>/dev/null || findmnt -S "${_dev_path}" &>/dev/null; then
        echo "Skip ${_label}: ${_dev_path} is already mounted"
        return 0
    fi

    mkdir -p "${_mnt_dir}"
    echo "Mounting ${_label} to ${_mnt_dir}"
    # sudo mount "${_dev_path}" "${_mnt_dir}"
    # add noatime,nodiratime to the mount options
    sudo mount -o noatime,nodiratime,data=ordered,nodelalloc,nolazytime "${_dev_path}" "${_mnt_dir}"
    # give permission to the mount directory
    sudo chmod -R 777 "${_mnt_dir}"
}

mount_dev_if_needed "eMMC" "${EMMC_DEV_NAME}" "${EMMC_OFFLOAD_DIR}"
mount_dev_if_needed "NVMe" "${NVME_DEV_NAME}" "${NVME_OFFLOAD_DIR}"

echo "Done: Setting up disk"
echo "--------------------------------"

##########################################################

echo "Starting to check model weights..."

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

echo "Done: Making np weights at $MODEL_PATH_BASE"
echo "--------------------------------"

##########################################################

echo "Checking remaining disk space..."

_GIB=$((1024 * 1024 * 1024))
_EMMC_MIN_FREE=$((64 * _GIB))
_NVME_MIN_FREE=$((200 * _GIB))

# Returns 0 if / is on the given device name (e.g. mmcblk0p1), same logic as mount_dev_if_needed.
root_fs_on_device() {
    local _dev_name="$1"
    local _dev_path="/dev/${_dev_name}"
    [[ -b "${_dev_path}" ]] || return 1
    local _target_real
    _target_real=$(readlink -f "${_dev_path}")
    local _root_src _root_real
    _root_src=$(findmnt -n -o SOURCE / 2>/dev/null | head -1)
    [[ -z "${_root_src}" ]] && return 1
    _root_real=""
    if [[ "${_root_src}" == "/dev/root" ]]; then
        _root_real=$(readlink -f /dev/root)
    elif [[ -b "${_root_src}" ]]; then
        _root_real=$(readlink -f "${_root_src}")
    elif [[ "${_root_src}" == UUID=* ]]; then
        _root_real=$(blkid -U "${_root_src#UUID=}" 2>/dev/null || true)
        [[ -n "${_root_real}" ]] && _root_real=$(readlink -f "${_root_real}")
    fi
    [[ -n "${_root_real}" && "${_root_real}" == "${_target_real}" ]]
}

check_disk_free_space() {
    local _label="$1"
    local _dev_name="$2"
    local _mount_dir="$3"
    local _min_free_bytes="$4"
    local _dev_path="/dev/${_dev_name}"

    if [[ ! -b "${_dev_path}" ]]; then
        echo "Error: ${_label} block device ${_dev_path} not found."
        exit 1
    fi

    if [[ ! -d "${_mount_dir}" ]]; then
        echo "Error: ${_label} mount directory ${_mount_dir} does not exist."
        exit 1
    fi

    # Allow: (1) ${_mount_dir} is its own mount point, or (2) it lives on / and / is on ${_dev_path}.
    if findmnt -n "${_mount_dir}" &>/dev/null; then
        :
    elif root_fs_on_device "${_dev_name}"; then
        echo "  (${_label}: ${_mount_dir} is on root FS, same device as ${_dev_path} — using df on this path)"
    else
        echo "Error: ${_mount_dir} is not mounted and root is not on ${_dev_path} (cannot verify free space)."
        exit 1
    fi

    local _avail
    _avail=$(df -B1 "${_mount_dir}" 2>/dev/null | tail -1 | awk '{print $4}')
    if [[ -z "${_avail}" || ! "${_avail}" =~ ^[0-9]+$ ]]; then
        echo "Error: Could not read free space for ${_mount_dir}."
        exit 1
    fi

    if [[ "${_avail}" -lt "${_min_free_bytes}" ]]; then
        echo "Error: ${_label} free space on ${_mount_dir} is below minimum required."
        echo "  Device: ${_dev_path}"
        echo "  Available (bytes): ${_avail} (need >= ${_min_free_bytes})"
        exit 1
    fi

    local _avail_gib _need_gib
    _avail_gib=$(awk "BEGIN {printf \"%.2f\", ${_avail} / ${_GIB}}")
    if [[ "${_label}" == "eMMC" ]]; then
        _need_gib="64"
    else
        _need_gib="256"
    fi
    echo "  ${_label} OK: ${_dev_path} @ ${_mount_dir} — ${_avail_gib} GiB free (>= ${_need_gib} GiB free)"
}

check_disk_free_space "eMMC" "${EMMC_DEV_NAME}" "${EMMC_OFFLOAD_DIR}" "${_EMMC_MIN_FREE}"
check_disk_free_space "NVMe" "${NVME_DEV_NAME}" "${NVME_OFFLOAD_DIR}" "${_NVME_MIN_FREE}"

echo "Disk check done."
echo "--------------------------------"

##########################################################

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

echo "Checking power mode and jetson clocks..."
check_powermode
check_jetson_clocks
echo "Done: Checking power mode and jetson clocks"
echo "--------------------------------"

##########################################################

echo "Done: Setup"
echo "--------------------------------"
 
echo "Now you can start the evaluation in 'run_evaluation.ipynb'."
echo "--------------------------------"

