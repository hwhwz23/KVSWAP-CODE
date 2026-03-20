
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



## 2. Run on Your Own Device

### 2.1 Requirements

#### Hardware

The **NVIDIA Jetson Orin AGX** has built-in **eMMC** storage; please additionally attach an **NVMe SSD** via the onboard PCIe interface.

Since KV cache data will be offloaded to eMMC and NVMe, please ensure:

- **eMMC free space**: >= 64GB
- **NVMe free space**: >= 200GB

To save eMMC space, please install the system image on the NVMe SSD.

#### Software

- **JetPack**: 6.2
- **Linux Kernel**: 5.15.148-tegra
- **CUDA**: 12.6
- **Python**: 3.10
- **PyTorch**: 2.7 (with CUDA)


### 2.2 Setup

After confirming that your device meets the hardware and software requirements above, proceed with the setup steps below.

uv


git clone this-repo/engine
TODO...


### 2.3 How to Run

All instructions and scripts are provided in `run_evaluation.ipynb`.
Please open this notebook on your device and execute the cells in order.

