This directory stores prebuilt wheels used by scripts/setup.sh when installing the environment.

Included wheels (Python 3.10, linux_aarch64 / Jetson):

  torch-2.7.0-cp310-cp310-linux_aarch64.whl
  flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl
  triton-3.2.0-cp310-cp310-linux_aarch64.whl
  vllm-0.10.1.dev271+g60523a731.cu126-cp310-cp310-linux_aarch64.whl

Torch, FlashAttention, and Triton — download compatible builds from the Jetson AI Lab index:

  https://pypi.jetson-ai-lab.io/

The vLLM wheel is a custom KVSwap build: it extends vLLM 0.10.1 by widening the paged-attention 
block sizes in csrc/attention/paged_attention_v1.cu and csrc/attention/paged_attention_v2.cu.

Obtain that wheel from:

  https://huggingface.co/zhww/Wheels4KVSwap

Put all downloaded .whl files in this folder (engine/wheel_pkgs) before running setup.
