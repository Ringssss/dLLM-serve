# dLLM-serve: Frontier-Guided Forward-Pass Scheduling for Diffusion LLM Serving

**P90 latency -82% | Throughput +104% | TPOT -47%** on SGLang with LLaDA2.0-mini (H100, TP=1/2/4).

Frontier-guided dLLM serving optimizations: **Admission → Execution → Termination**, each driven by the denoising frontier. Drop-in replacement for SGLang's default `LowConfidence` algorithm. Zero impact on AR model paths.

## Results (Real SGLang Online Serving)

### CW-SRPT V3 vs LowConfidence

| TP | Rate | TPS Improvement | P90 Latency Reduction |
|----|------|----------------|----------------------|
| 1 | 5/s | +9% | **-69%** |
| 1 | 10/s | **+92%** | **-82%** |
| 1 | 20/s | **+104%** | -67% |
| 2 | 10/s | +45% | **-81%** |
| 2 | 20/s | **+101%** | -73% |
| 4 | 10/s | +27% | **-76%** |
| 4 | 20/s | **+93%** | -76% |

Single request TPOT: **3.3→2.8ms (TP=1)**, **2.8→2.2ms (TP=2)**, **2.1→1.9ms (TP=4)**

Output quality: **32/32 readable** across HumanEval, GSM8K, MGSM, MT-Bench ✅

## Three-Pronged Architecture

| Stage | Mechanism | Effect |
|-------|-----------|--------|
| **Admission** | Frontier-aware slot allocation with aging | P90/P99 tail latency reduction |
| **Execution** | Adaptive denoising stride (target-based top-k) | Forward passes per block reduced |
| **Termination** | Active-set early break + scheduler compaction | Batch slot utilization improvement → TPS doubled |

## Quick Start

### Step 1: Install

```bash
git clone https://github.com/Ringssss/dLLM-serve.git
cd dLLM-serve
bash install.sh
```

### Step 2: Launch

```bash
# V3 (recommended — latest three-pronged architecture)
python -m sglang.launch_server \
    --model-path inclusionAI/LLaDA2.0-mini \
    --dllm-algorithm CW_SRPT_V3 \
    --max-running-requests 8 \
    --disable-radix-cache \
    --trust-remote-code \
    --port 30100

# TP=2
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path inclusionAI/LLaDA2.0-mini \
    --dllm-algorithm CW_SRPT_V3 \
    --max-running-requests 8 \
    --disable-radix-cache \
    --trust-remote-code \
    --port 30100 \
    --tp-size 2
```

### Step 3: Send Requests

```bash
curl -X POST http://localhost:30100/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-mini",
    "prompt": "Write a Python function that checks if a number is prime.",
    "max_tokens": 128,
    "temperature": 0
  }'
```

### Step 4: Benchmark

```bash
# Quick benchmark
python benchmarks/bench_v3.py --tp 1 --reqs 64 --rates 5,10,20

# Full comparison: LowConfidence vs V2 vs V3
python benchmarks/bench_v3.py --tp 1 --reqs 64 --rates 5,10,20 --algos LowConfidence,CW_SRPT_V2,CW_SRPT_V3

# TP sweep
python benchmarks/bench_v3.py --tp 2 --reqs 64
python benchmarks/bench_v3.py --tp 4 --reqs 64

# Quality check only
python benchmarks/bench_v3.py --quality-only --algos CW_SRPT_V3
```

### Uninstall

```bash
bash scripts/uninstall.sh
```

## How It Works

### Contribution 1: Frontier-Aware Admission (`scheduler.py`)

Replaces FIFO request admission with frontier-score-based slot allocation:

```python
score = remaining_work / (1 + aging_factor * wait_rounds)
```

Requests close to completion get priority. Aging prevents starvation.

### Contribution 2: Adaptive Denoising Stride (`cw_srpt_v3.py`)

Instead of fixed threshold ("unmask everything above 0.95"), computes a target stride:

```python
stride = ceil(n_masked / remaining_iters)  # "how many tokens should I unmask?"
```

Easy blocks take larger strides; stalled blocks get a minimum progress guarantee.

### Contribution 3: Active-Set Early Break (`cw_srpt_v3.py`)

When >50% of blocks in a batch are done, break early and return to scheduler. The scheduler compacts done blocks out and refills with new requests — true dLLM-native continuous batching.

### Frontier Writeback (`scheduler.py`)

After each batch result, the scheduler reads `n_masked` and `confidence` back from the completed forward pass and writes them to `Req` objects. This closes the algorithm→scheduler information loop.

## Why dLLM-Only?

| Property | AR LLM | dLLM |
|----------|--------|------|
| Per-token confidence | ❌ | ✅ — enables frontier-aware scheduling |
| Preemption cost | ~100MB KV cache swap | **~2KB** mask state |
| Execution control | Fixed (1 token/step) | **Adaptive stride** (1-16 tokens/step) |
| Block structure | No | **Yes** — enables active-set compaction |

All changes are inside `dllm/` paths. AR models are **not affected**.

## Repository Structure

```
dLLM-serve/
├── install.sh                              # One-command setup
├── cwsrpt_v2/                              # Core: SGLang patches
│   ├── algorithm/
│   │   ├── cw_srpt_v3.py                  # V3: stride + early break + writeback (LATEST)
│   │   ├── cw_srpt_v2.py                  # V2: adaptive threshold + vectorized
│   │   └── cw_srpt.py                     # V1: basic adaptive threshold
│   └── mixin/
│       ├── req.py                          # Frontier state fields + remaining_work
│       └── scheduler.py                    # Frontier writeback + frontier admission + aging
├── benchmarks/
│   ├── bench_v3.py                         # Standard benchmark (TP sweep + quality)
│   ├── diffserve_v2.py                     # Early experiment: continuous batching
│   └── diffserve_v3.py                     # Early experiment: CW-SRPT prototype
├── ppopp/                                  # Paper figures: bench + plot scripts
│   ├── bench_fig9_100B*.py                 # Benchmark scripts for paper figures
│   ├── bench_fig13_quality.py              # Quality evaluation
│   ├── bench_100B_tp4.py                   # TP=4 100B benchmark
│   ├── collect_fig*.py                     # Data collection
│   ├── plot_fig*.py                        # Plotting scripts
│   ├── fig*_raw_data.json                  # Raw data
│   └── fig*.pdf / fig*.png                 # Generated figures
├── diffserve/                              # Standalone serving framework
├── scripts/
│   ├── bench.sh                            # Shell benchmark wrapper
│   └── uninstall.sh                        # Restore original SGLang
├── docs/
│   ├── diffserve_confidence_aware_scheduling.md  # Full experiment log (12 experiments)
│   ├── cwsrpt_v3_sglang_incremental_code.md      # Complete incremental code
│   └── DESIGN_SPECDIFF.md                        # SpecDiff design doc
└── README.md
```

## Full Results

### TP=1 (64 reqs, max_tokens=128, H100 80GB)

| Rate | Metric | LowConfidence | CW-SRPT V3 | Delta |
|------|--------|-------------|-----------|-------|
| 5/s | TPS | 603 | 659 | +9% |
| 5/s | P90 | 2511ms | 771ms | -69% |
| 10/s | TPS | 629 | 1207 | **+92%** |
| 10/s | P90 | 6875ms | 1213ms | **-82%** |
| 20/s | TPS | 655 | 1338 | **+104%** |
| 20/s | P90 | 8970ms | 2983ms | -67% |

### TP=2

| Rate | Metric | LowConfidence | CW-SRPT V3 | Delta |
|------|--------|-------------|-----------|-------|
| 5/s | P90 | 1297ms | 514ms | -60% |
| 10/s | TPS | 857 | 1242 | +45% |
| 10/s | P90 | 3474ms | 655ms | **-81%** |
| 20/s | TPS | 872 | 1758 | **+101%** |
| 20/s | P90 | 6125ms | 1638ms | -73% |

### TP=4

| Rate | Metric | LowConfidence | CW-SRPT V3 | Delta |
|------|--------|-------------|-----------|-------|
| 5/s | P90 | 979ms | 393ms | -60% |
| 10/s | TPS | 1024 | 1301 | +27% |
| 10/s | P90 | 2115ms | 497ms | **-76%** |
| 20/s | TPS | 1068 | 2057 | **+93%** |
| 20/s | P90 | 4480ms | 1090ms | -76% |

## Evolution: V1 → V2 → V3

| Version | Key Mechanism | Best P90 | Best TPS |
|---------|-------------|---------|---------|
| V1 | Adaptive threshold (per-block loop) | -65% | +59% |
| V2 | + Vectorized block + early exit | -68% | +64% |
| **V3** | + Stride controller + early break + frontier admission | **-82%** | **+104%** |

## License

Apache 2.0
