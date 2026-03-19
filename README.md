# KVSWAP-CODE

## 1. Overview

This repository contains the code for the artifact evaluation of our paper **“KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference”**.

## 2. Abstract

Language models (LMs) underpin emerging mobile and embedded AI applications like meeting and video summarization and document analysis, which often require processing multiple long-context inputs. Running an LM locally on-device improves privacy, enables offline use, and reduces cost, but long-context inference quickly hits a memory capacity wall as the key-value (KV) cache grows linearly with context length and batch size. Existing KV-cache offloading schemes are designed to transfer cache data from GPU memory to CPU memory; however, they are not suitable for embedded and mobile systems, where the CPU and GPU (or NPU) typically share a unified memory and the non-volatile secondary storage (disk) offers limited I/O bandwidth. We present KVSwap, a software framework tailored for local devices that achieves high memory efficiency while effectively leveraging disk storage. KVSwap stores the full cache on disk, uses highly compact in-memory metadata to predict which entries to preload, overlaps computation with hardware-aware disk access, and orchestrates read patterns to match storage device characteristics. Our evaluation shows that across representative LMs and storage types, KVSwap delivers higher throughput under tight memory budgets while maintaining generation quality over existing KV cache offloading schemes. 

## 3. Evaluation

This artifact provides two evaluation tracks:

- **Accuracy evaluation**: run on a high-end server; see `quality/README.md` for the full guide.
- **Throughput evaluation**: run on NVIDIA Jetson Orin AGX; see `engine/README.md` for the full guide..

