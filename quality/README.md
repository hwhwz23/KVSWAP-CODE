
# Accuracy Evaluation

> **Recommended for AE reviewers — remote evaluation (no local setup).**  
> For the fastest accuracy evaluation, we provide **remote access** to a **pre-configured GPU server**. You can run the workflow **in a web browser** without installing anything locally.  
> **Details:** see [`remote_evaluation.ipynb`](remote_evaluation.ipynb).

If you would like to run the evaluation on your own GPU server, please follow the instructions below.

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

## 2. Setup

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

### Setup Environment

```bash
# Clone only the `quality` directory and skip LFS objects initially.
git clone --filter=blob:none --no-checkout https://github.com/hwhwz23/KVSWAP-CODE.git
cd ./KVSWAP-CODE
git sparse-checkout init --cone
git sparse-checkout set quality
GIT_LFS_SKIP_SMUDGE=1 git checkout main
cd ./quality/
# Install dependencies
bash scripts/install.sh
```

## 3. How to Run

### 3.1 Prepare API key

Some evaluations (e.g., NIAH and MLVU) require a judge LLM to score model outputs. By default, we use the `deepseek-chat` model as the judge. Please set your Deepseek API key in the environment:

```bash
export DS_API_KEY="your_deepseek_api_key"
```

If you are an artifact-evaluation reviewer, we can provide an API key upon request via HotCRP.

### 3.2 Quick Evaluation

To quickly obtain evaluation results, we provide a **quick mode** that randomly samples a subset of the benchmark data (about **30%** by default) and skips 14B models. This significantly reduces runtime while preserving overall result trends. On our machine, a complete quick run takes approximately **35 hours**.

**All generated results will be stored under the `RESULTS/` directory.**

#### 3.2.1 Data preparation

First, download benchmark data and model weights. This process typically takes around **1-2 hours**, depending on your network and disk.

```bash
# Download models
bash ./scripts/download_model.sh
# Download MLVU dataset
bash ./scripts/download_dataset.sh
```

#### 3.2.2 Step-by-step quick run

##### Figure 9 (1 hour)

```bash
# Tag used to store your evaluation results
export EVAL_USER="test0"
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/fig-9.sh
```

##### Table 2 (3-4 hours)

```bash
# Tag used to store your evaluation results
export EVAL_USER="test0"
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-2.sh
```

##### Table 3 Left (24 hours)

**In quick mode, we do not evaluate Qwen3-14B because it is very time-consuming.**

```bash
# Tag used to store your evaluation results
export EVAL_USER="test0"
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-3-left.sh
```

##### Table 3 Right (4-5 hours)

**In quick mode, we do not evaluate InternVL3-14B because it is very time-consuming.**

```bash
# Tag used to store your evaluation results
export EVAL_USER="test0"
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-3-right.sh
```

##### Accuracy results in Figure 11 (1 hour)

**This figure reuses part of the Table 2 results. Please run Table 2 before running this script.**
Here only covers accuracy. For the throughput part of Figure 11, see the throughput evaluations in the repository's `engine/` directory.

```bash
# Tag used to store your evaluation results
export EVAL_USER="test0"
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/fig-11-acc.sh
```

#### 3.2.3 Alternative: Single-step quick run (35 hours)

To run all quick evaluations at once, execute the following script. It runs all step-by-step scripts in sequence. If you have already completed the step-by-step runs above, you do not need to run this again.

```bash
# Tag used to store your evaluation results
export EVAL_USER="test0"
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/quick_run.sh
```

### 3.3 Optional: Full Evaluation

This mode evaluates **all models** on the **full benchmark datasets** and may take substantially longer to complete.

#### 3.3.1 Data preparation

First, download full benchmark data and model weights. 

```bash
# Download models
bash ./scripts/download_model.sh full
# Download MLVU dataset
bash ./scripts/download_dataset.sh full
```

#### 3.3.2 Run

To evaluate on the full benchmark dataset and all models, run:

```bash
# Tag used to store your evaluation results
export EVAL_USER="test0"
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/full_run.sh
```

### 4. Optional: Prepare your own adapters

We provide pre-built adapter weights under the `adapters/` directory, and the evaluation scripts will automatically load and use them when applicable.

To build adapters yourself, refer to `scripts/prepare_adapters.sh` as an example.