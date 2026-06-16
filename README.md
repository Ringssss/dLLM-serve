# dLLM-serve: CW-SRPT V2 — dLLM-Native Serving Optimizations

**P90 latency -68% | Throughput +64% | TPOT -51%** on SGLang with LLaDA2.0-mini (H100, TP=1/2/4).

Three dLLM-specific optimizations for diffusion LLM online serving. Drop-in replacement for SGLang's default `LowConfidence` algorithm. Zero impact on AR model paths.

## Results (Real SGLang Online Serving)

| TP | Rate | TPS Improvement | P90 Latency Reduction |
|----|------|----------------|----------------------|
| 1 | 10/s | +59% | **-65%** |
| 2 | 10/s | +44% | **-68%** |
| 2 | 20/s | **+64%** | -57% |
| 4 | 10/s | +26% | **-65%** |
| 4 | 20/s | +53% | -53% |

Single request TPOT: **3.3→1.7ms (TP=1)**, **2.6→1.3ms (TP=2)**, **2.1→1.1ms (TP=4)**

## Setup (5 minutes)

### Prerequisites

- NVIDIA GPU (H100/A100 recommended)
- CUDA 12.x
- Python 3.10+
- SGLang (with dLLM support)
- A dLLM model (e.g., [LLaDA2.0-mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini))

### Step 1: Install SGLang

```bash
pip install "sglang[all]>=0.5.0"
# Or from source:
# git clone https://github.com/sgl-project/sglang.git
# cd sglang && pip install -e "python[all]"
```

### Step 2: Install CW-SRPT V2

```bash
git clone https://github.com/Ringssss/dLLM-serve.git
cd dLLM-serve
bash install.sh
```

The install script automatically:
- Detects your SGLang installation
- Backs up original files
- Copies the CW-SRPT V2 algorithm + scheduler patches

### Step 3: Verify

```bash
python -c "from sglang.srt.dllm.algorithm.cw_srpt_v2 import CW_SRPT_V2; print('✅ CW-SRPT V2 installed')"
```

## Usage

### Launch Server

```bash
# CW-SRPT V2 (our method)
python -m sglang.launch_server \
    --model-path inclusionAI/LLaDA2.0-mini \
    --dllm-algorithm CW_SRPT_V2 \
    --max-running-requests 8 \
    --disable-radix-cache \
    --trust-remote-code \
    --port 30100

# With TP=2
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path inclusionAI/LLaDA2.0-mini \
    --dllm-algorithm CW_SRPT_V2 \
    --max-running-requests 8 \
    --disable-radix-cache \
    --trust-remote-code \
    --port 30100 \
    --tp-size 2
```

### Send Requests

```bash
curl -X POST http://localhost:30100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-mini",
    "prompt": "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Write a Python function that checks if a number is prime.<|role_end|><role>ASSISTANT</role>",
    "max_tokens": 128,
    "temperature": 0
  }'
```

### Run Benchmark

```bash
# Compare LowConfidence (baseline) vs CW-SRPT V2
bash scripts/bench.sh --tp 1 --reqs 64 --model /path/to/LLaDA2.0-mini

# TP=2
bash scripts/bench.sh --tp 2 --reqs 64

# TP=4
bash scripts/bench.sh --tp 4 --reqs 64
```

### Restore Original (Uninstall)

```bash
bash scripts/uninstall.sh
```

## How It Works

### Three Contributions

**1. Fused Adaptive Threshold** (`cwsrpt_v2/algorithm/cw_srpt_v2.py`)

Standard dLLM uses fixed threshold (e.g., 0.95) for unmasking tokens. We make it adaptive:
- High mean confidence → lower threshold → unmask more tokens per iteration
- Minimum 2 tokens unmasked per iteration (not just 1)
- Stall decay: stuck blocks get progressively more aggressive

This cuts forward passes per block from ~15 to ~8, reducing TPOT by ~50%.

**2. Vectorized Block Processing** (`cwsrpt_v2/algorithm/cw_srpt_v2.py`)

Replace Python per-block for-loop with single vectorized PyTorch operation:
```python
# Before (LowConfidence): O(batch_size) Python iterations
for batch_id in range(batch_size):
    # per-block threshold logic...

# After (CW-SRPT V2): O(1) vectorized
logits_view = full_logits.view(batch_size, block_size, -1)
x0 = logits_view.argmax(dim=-1)  # all blocks at once
```

**3. CW-SRPT Priority Queue** (`cwsrpt_v2/mixin/scheduler.py`)

The scheduler sorts decode requests by confidence-weighted remaining work:
```python
priority = n_masked * (1 - mean_confidence)
```
Requests close to completion (high confidence, few masked tokens) run first, finishing faster and freeing batch slots for waiting requests.

### Why dLLM-Only?

These optimizations exploit properties unique to diffusion LLMs:
- **Per-token confidence**: AR LLMs don't have confidence at masked positions
- **Block structure**: Fixed 32-token blocks enable vectorized processing
- **Free preemption**: dLLM state is ~2KB (masked tokens), not ~100MB (KV cache)

AR models are **not affected** — all changes are inside `dllm/` paths that only activate with `--dllm-algorithm`.

## Repository Structure

```
dLLM-serve/
├── install.sh                          # One-command setup
├── cwsrpt_v2/                          # Core: SGLang patches
│   ├── algorithm/
│   │   ├── cw_srpt_v2.py              # V2 algorithm (main contribution)
│   │   └── cw_srpt.py                 # V1 algorithm
│   └── mixin/
│       ├── req.py                      # Confidence tracking fields
│       └── scheduler.py                # CW-SRPT priority queue
├── diffserve/                          # Standalone serving framework
│   ├── engine.py                       # KV cache + adaptive threshold engine
│   ├── scheduler.py                    # 5 scheduling policies
│   ├── bench_comparison.py             # A/B benchmark
│   └── ...
├── scripts/
│   ├── bench.sh                        # Run full benchmark
│   └── uninstall.sh                    # Restore originals
├── benchmarks/                         # Experiment scripts
├── docs/
│   ├── diffserve_confidence_aware_scheduling.md  # Full experiment log (12 experiments)
│   └── DESIGN_SPECDIFF.md             # SpecDiff design doc
└── README.md
```

## Full Results Table

### TP=1 (64 reqs, max_tokens=128, H100 80GB)

| Rate | Metric | LowConfidence | CW-SRPT V2 | Delta |
|------|--------|-------------|-----------|-------|
| 5/s | TPS | 597 | 645 | +8% |
| 5/s | P90 | 2649ms | 1036ms | -61% |
| 10/s | TPS | 626 | 994 | +59% |
| 10/s | P90 | 6930ms | 2429ms | -65% |
| 20/s | TPS | 653 | 1008 | +54% |
| 20/s | P90 | 9000ms | 4886ms | -46% |

### TP=2

| Rate | Metric | LowConfidence | CW-SRPT V2 | Delta |
|------|--------|-------------|-----------|-------|
| 5/s | P90 | 1341ms | 681ms | -49% |
| 10/s | TPS | 854 | 1234 | +44% |
| 10/s | P90 | 3501ms | 1125ms | **-68%** |
| 20/s | TPS | 868 | 1421 | **+64%** |
| 20/s | P90 | 6170ms | 2671ms | -57% |

### TP=4

| Rate | Metric | LowConfidence | CW-SRPT V2 | Delta |
|------|--------|-------------|-----------|-------|
| 5/s | P90 | 951ms | 544ms | -43% |
| 10/s | TPS | 1017 | 1281 | +26% |
| 10/s | P90 | 2164ms | 754ms | -65% |
| 20/s | TPS | 1058 | 1619 | +53% |
| 20/s | P90 | 4548ms | 2115ms | -53% |

## License

Apache 2.0
