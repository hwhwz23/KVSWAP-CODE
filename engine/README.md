# Throughput Evaluation

We use **NVIDIA Jetson Orin AGX** as the target platform and provide two ways to run the throughput evaluation:

1. **Remote Access**: connect to our pre-configured platform and run the evaluation directly.
2. **Local Execution**: run the evaluation on your own device using the provided scripts and code.


## 1. Remote Access

This option is primarily intended for artifact-evaluation reviewers. No setup is required. You can open a Jupyter Notebook in your **browser** and run the evaluation remotely.

After opening the remote environment in your browser:

1. Open `run_evaluation.ipynb`.
2. Execute all cells in order.

For security reasons, we do not publish the remote access IP address or password in this repository.  
Please contact us via HotCRP, and we will share the credentials promptly.


## 2. Run on your own device

### 2.1 Requirements

#### Hardware

The **NVIDIA Jetson Orin AGX** includes **eMMC**; attach an **NVMe SSD** over the onboard PCIe slot.

KV-cache data is offloaded to eMMC and NVMe. Ensure:

- **eMMC**: ≥ 64 GB  
- **NVMe**: ≥ 256 GB  

To reduce pressure on eMMC, we recommend **installing the system image on the NVMe SSD**.

#### Software

- **JetPack**: 6.2  
- **Linux kernel**: 5.15.148-tegra  
- **CUDA**: 12.6  
- **Python**: 3.10  
- **PyTorch**: 2.7  


### 2.2 Setup

Confirm your hardware and software match the requirements above, then follow the steps below.

#### Install `uv`

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (or use the one-liner):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Clone the repository (sparse checkout: `engine` only)

```bash
git clone --filter=blob:none --no-checkout https://github.com/hwhwz23/KVSWAP-CODE.git
cd KVSWAP-CODE
git sparse-checkout init --cone
git sparse-checkout set engine
GIT_LFS_SKIP_SMUDGE=1 git checkout main
cd engine
```

#### Download wheel packages

Download the required `.whl` files and place them under `wheel_pkgs/`.  
See [`wheel_pkgs/readme.txt`](wheel_pkgs/readme.txt) for details.

#### User configuration

Set paths and evaluation options to match your machine (block device names, mount points, model locations, log root, and a short **user id** for result folders):

```bash
# Replace with your own settings
export EMMC_DEV_NAME='mmcblk0p1'
export EMMC_OFFLOAD_DIR='/mnt/emmc/offload'
export NVME_DEV_NAME='nvme0n1'
export NVME_OFFLOAD_DIR='/mnt/nvme/offload'
export MODEL_PATH_BASE_HF='../../ext_disk/model_weights_hf'
export MODEL_PATH_BASE='../../ext_disk/model_weights'
export EVAL_LOG_DIR='../../ext_disk/kvswap_logs'

export EVAL_USER='test0'   # **Change this** identifier for storing results
export EVAL_MODE='quick' # or 'full'
```

#### Set up the environment

This installs dependencies and performs hardware checks:

```bash
bash ./scripts/setup.sh
```


### 2.3 How to run

Open `run_evaluation.ipynb` on the device and execute the cells **in order**. All evaluation steps and commands are documented there.
