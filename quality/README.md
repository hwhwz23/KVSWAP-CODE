
# Accuracy Evaluation


## 1. Requirements

### 1.1 Hardware

We run accuracy experiments on a high-end server. Recommended configuration:
- **CPU**: X86_64
- **GPU**: NVIDIA A100 (80GB)
- **Host memory**: >= 128GB
- **Disk**: >= 500GB free space

### 1.2 Software

- **OS**: Ubuntu 22.04
- **CUDA**: 12.6
- **Python**: 3.10

And make sure the following tools are installed:
- `uv`
- `git-lfs`

#### Install `uv`

Run the following command, or refer to `https://docs.astral.sh/uv/getting-started/installation/`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install `git-lfs`

Run the following commands, or refer to `https://graphite.com/guides/how-to-install-git-lfs-on-ubuntu`:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs
git lfs install
```




## 2. Setup and Data Download

This process typically takes around **XXX hours**, depending on your network and disk.

```bash
# Clone the repo first, but skip LFS objects initially.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/hwhwz23/KVSWAP-CODE.git
cd ./KVSWAP-CODE/quality/
# Use GPU 0 for evaluation.
export CUDA_VISIBLE_DEVICES="0"
# Install dependencies, download models, and download benchmark datasets.
bash scripts/init.sh
```

## 3. How to Run

### 3.1 Prepare API key

Some evaluations (e.g., NIAH and MLVU) require a judge LLM to score model outputs. By default, we use the `deepseek-chat` model as the judge. Please set your Deepseek API key in the environment:

```bash
export DS_API_KEY="your_deepseek_api_key"
```

If you are an artifact-evaluation reviewer, we can provide an API key upon request via HotCRP.

### 3.2 Quick Evaluation

To quickly obtain evaluation results, we provide a **quick mode** that randomly samples a subset of the benchmark data (about **30%** by default). This significantly reduces runtime while preserving the overall trends of the results. On our machine, a complete quick run takes approximately **XXX hours**.

**All generated results will be stored under the `RESULTS/` directory.**


#### 3.2.1 Step-by-step quick run

##### 3.2.1.1 Figure 9

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/fig-9.sh
```

##### 3.2.1.2 Table 2

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-2.sh
```

##### 3.2.1.3 Table 3 (left)

**In quick mode, we do not evaluate Qwen3-14B because it is very time-consuming.**

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-3-left.sh
```

##### 3.2.1.4 Table 3 (right)

**In quick mode, we do not evaluate InternVL3-14B because it is very time-consuming.**

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-3-right.sh
```

##### 3.2.1.5 Accuracy results in Figure 11

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/fig-11-acc.sh
```


#### 3.2.2 Alternative: Single-step quick run

To run all quick evaluations at once:
```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/quick_run.sh
```


### 3.3 Optional: Full Evaluation

This mode evaluates the **full benchmark dataset** and can take a long time to complete. On our machine, a full run takes approximately **XXX hours**.

To evaluate on the full benchmark dataset, run:

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/full_run.sh
```


### 3.4 Optional: Prepare your own adapters

We provide pre-built adapter weights under the `adapters/` directory, and the evaluation scripts will automatically load and use them when applicable.

Optionally, you can also build your own adapters. Please refer to the instructions below.

**TODO**: 
