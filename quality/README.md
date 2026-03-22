
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

To quickly obtain evaluation results, we provide a **quick mode** that randomly samples a subset of the benchmark data (about **30%** by default) and skips 14B models. This significantly reduces runtime while preserving overall result trends. On our machine, a complete quick run takes approximately **XXX hours**.

**All generated results will be stored under the `RESULTS/` directory.**

#### 3.2.1 Data preparation (XX hours)

First, download adapter weights, a subset of benchmark data and model weights. This process typically takes around **XXX hours**, depending on your network and disk.

```bash
# Download adapter weights
git lfs pull --include="adapters/**/*.pt"
# Download models
bash ./scripts/download_model.sh
# Download datasets
bash ./scripts/download_dataset.sh
```

#### 3.2.2 Step-by-step quick run

##### Figure 9 (XX hours)

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/fig-9.sh
```

##### Table 2 (XX hours)

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-2.sh
```

##### Table 3 Left (XX hours)

**In quick mode, we do not evaluate Qwen3-14B because it is very time-consuming.**

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-3-left.sh
```
 
##### Table 3 Right (XX hours)

**In quick mode, we do not evaluate InternVL3-14B because it is very time-consuming.**

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/tab-3-right.sh
```

##### Accuracy results in Figure 11 (XX hours)

**This figure reuses part of the Table 2 results. Please run Table 2 before running this script.**

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/fig-11-acc.sh
```


#### 3.2.3 Alternative: Single-step quick run (XX hours)

To run all quick evaluations at once, execute the following script. It runs all step-by-step scripts in sequence. If you have already completed the step-by-step runs above, you do not need to run this again.

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/quick_run.sh
```


### 3.3 Optional: Full Evaluation

This mode evaluates the **full benchmark dataset** and can take a long time to complete. On our machine, a full run takes approximately **XXX hours**.


#### 3.3.1 Data preparation (XX hours)

First, download adapter weights, full benchmark data and model weights. This process typically takes around **XXX hours**, depending on your network and disk.

```bash
# Download adapter weights
git lfs pull --include="adapters/**/*.pt"
# Download models
bash ./scripts/download_model.sh full
# Download datasets
bash ./scripts/download_dataset.sh full
```


#### 3.3.2 Run (XX hours)

To evaluate on the full benchmark dataset and models, run:

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
bash ./scripts/full_run.sh
```


### 4. Optional: Prepare your own adapters

We provide pre-built adapter weights under the `adapters/` directory, and the evaluation scripts will automatically load and use them when applicable.

Optionally, you can also build your own adapters. Please refer to the instructions below.

**TODO**: 
