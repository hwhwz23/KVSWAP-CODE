# KVSWAP-CODE

## 1. Overview

This repository contains the code for the artifact evaluation of our paper **“KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference”**.

## 2. Abstract

Language models (LMs) underpin emerging mobile and embedded AI applications like meeting and video summarization and document analysis, which often require processing multiple long-context inputs. Running an LM locally on-device improves privacy, enables offline use, and reduces cost, but long-context inference quickly hits a memory capacity wall as the key-value (KV) cache grows linearly with context length and batch size. Existing KV-cache offloading schemes are designed to transfer cache data from GPU memory to CPU memory; however, they are not suitable for embedded and mobile systems, where the CPU and GPU (or NPU) typically share a unified memory and the non-volatile secondary storage (disk) offers limited I/O bandwidth. We present KVSwap, a software framework tailored for local devices that achieves high memory efficiency while effectively leveraging disk storage. KVSwap stores the full cache on disk, uses highly compact in-memory metadata to predict which entries to preload, overlaps computation with hardware-aware disk access, and orchestrates read patterns to match storage device characteristics. Our evaluation shows that across representative LMs and storage types, KVSwap delivers higher throughput under tight memory budgets while maintaining generation quality over existing KV cache offloading schemes. 

## 3. Evaluation

This artifact provides two evaluation tracks:

- **Accuracy evaluation**: run on a high-end server; see `quality/README.md` for the full guide.
- **Throughput evaluation**: run on NVIDIA Jetson Orin AGX; instructions and code are under `engine/` (local) and `engine_remote/` (remote access).

### 3.1 Accuracy evaluation

#### 3.1.1 Hardware requirements

We run accuracy experiments on a high-end server. Recommended configuration:
- **CPU**: X86_64
- **GPU**: NVIDIA A100 (80GB)
- **Host memory**: >= 128GB
- **Disk**: >= 500GB free space

#### 3.1.2 Software requirements

- **OS**: Ubuntu 22.04
- **CUDA**: 12.6
- **Python**: 3.10

#### 3.1.3 How to run

For detailed steps, please refer to `quality/README.md`.

### 3.2 Throughput evaluation

#### 3.2.1 Hardware requirements

We run throughput experiments on **NVIDIA Jetson Orin AGX**. The AGX has built-in **eMMC** storage; we additionally attach an **NVMe SSD**.

Since KV cache data will be offloaded to eMMC and NVMe, please ensure:

- **eMMC free space**: >= 64GB
- **NVMe free space**: >= 64GB

To save eMMC space, we install the system image on the NVMe SSD.

#### 3.2.2 Software requirements

- **JetPack**: 6.2
- **CUDA**: 12.6
- **Python**: 3.10
- **Pytorch**: 2.7 (with CUDA)

#### 3.2.3 How to run

We provide two ways to run throughput evaluation:

1. **Remote access via Jupyter Notebook**: refer to `engine_remote/` to access our pre-configured hardware platform.
2. **Local run (full code + instructions)**: refer to `engine/` to run everything on your own device (this requires that your device meet the hardware and software requirements above).
