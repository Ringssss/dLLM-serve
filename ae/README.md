# PAS artifact: public end-to-end reproduction

Artifact state: commit `89fb84092` and later on branch `ppopp-artifacts`
(https://github.com/Ringssss/dLLM-serve). This README describes that state;
pin the commit you actually evaluated with `git rev-parse HEAD`.

**What this artifact reproduces:** the end-to-end latency/throughput comparison
of PAS against the SGLang LowConfidence baseline on LLaDA2.0-mini, across four
public evaluation sets and an Azure trace replay, on 2×A100. Direction and
saturation behaviour match the paper; absolute magnitudes are hardware-specific
and are not claimed to transfer to the paper's H100 setup.

Compare SGLang LowConfidence and PAS (`CW_SRPT_V3`) on the same LLaDA2.0-mini
BF16 checkpoint, GPU allocation and requests. The public artifact covers
controlled-load latency/throughput trends and **Azure trace-derived replay**.
**Kimi traces are confidential and cannot be publicly released. We therefore
use Azure as the reproducible trace example; Kimi is outside this public scope.**

## Requirements

Linux x86-64, Python 3.12, Git, and an NVIDIA driver supporting CUDA 13 (580 or
newer). Validated on two A100 SXM4 40 GB GPUs, TP=2, running methods sequentially
on the same pair. Provision approximately 128 GB host RAM and 120 GB free disk
for the 32.5 GB model, environment, caches and results. These are provisioning
recommendations rather than measured minima. Setup needs PyPI and Hugging Face;
no author-hosted machine, private credentials or paid judge is required.

## Clone and run

```bash
git clone --branch ppopp-artifacts https://github.com/Ringssss/dLLM-serve.git
cd dLLM-serve
bash ae/reproduce.sh smoke
bash ae/reproduce.sh all
```

`smoke` is the getting-started test, budgeted at about 30 minutes after downloads
and installation. `all` runs four stages: a controlled-load sweep, a four-dataset
load sweep, the Azure replay and a 200-example GSM8K quality check. Budget about
60–90 minutes of measurement on the tested GPU pair, plus a one-time install and
checkpoint download. Save `git rev-parse HEAD` with your run. For a frozen AE
submission, use its supplied commit or source archive.

The entry point creates `.venv-ae`, installs `ae/requirements.lock`, obtains
`inclusionAI/LLaDA2.0-mini`, verifies model files against `ae/model-files.json`,
and prepares bundled public inputs. The model's resolved Hugging Face revision
is recorded. If upstream model content changes, checksum verification fails;
use the matching `--revision` with `ae/prepare.py`.

Cached model/environment and device selection:

```bash
AE_MODEL=/path/to/LLaDA2.0-mini AE_VENV=/path/to/venv \
  AE_GPUS=0,1 AE_TP=2 AE_PORT=30174 bash ae/reproduce.sh all
```

Use `AE_OUTPUT=/path/to/new-results` to choose the output directory. Each run
uses a new directory. An occupied port fails immediately; startup is protected
by a lock and readiness checks verify the algorithm identity.

Manual setup is equivalent:

```bash
bash ae/setup.sh
source .venv-ae/bin/activate
python ae/prepare.py
# Or verify existing weights:
python ae/prepare.py --local-model --model /path/to/LLaDA2.0-mini
```

The authors validated setup in a second empty environment and local checkpoint
hashes. Hugging Face downloading could not be exercised from that host because
its connection was unavailable; model acquisition remains a public download
step. No model weights are stored in Git.

## Bundled public inputs

- `ae/datasets/gsm8k.jsonl`: all 1,319 GSM8K test examples from OpenAI repository
  commit `3101c7d5072418e28b9008a6636bde82a006892c`, with sequential IDs added.
  Source identity, SHA-256 and MIT license are included. Preparation verifies
  and copies the file to `ae/data/`; no dataset download is needed.
- `ae/datasets/{mgsm,humaneval,mt_bench}.jsonl`: 200, 100 and 80 records for the
  load sweep, with source, licence (CC-BY-4.0, MIT, Apache-2.0), SHA-256 and byte
  counts in `ae/datasets/EVALSETS.json` and the licence texts alongside.
- `ae/traces/azure.csv`: 44,102 requests from 2024-05-12 00:00–00:30 UTC in the
  public Azure conversation trace. Columns contain relative arrival time,
  input tokens and output tokens. Source and subset checksums, CC-BY 4.0 notice
  and attribution are included. Only this fixed window is needed for replay.

To regenerate Azure's window from the public full CSV:

```bash
curl -L --fail -o AzureLLMInferenceTrace_conv_1week.csv \
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv
python ae/prepare_traces.py --azure AzureLLMInferenceTrace_conv_1week.csv \
  --output /path/to/new-azure-window
```

## Controlled load sweep

```bash
bash ae/reproduce.sh quick
source .venv-ae/bin/activate
python ae/run.py --model ae/models/LLaDA2.0-mini --profile quick \
  --rates 0.5 1 2 4 8 --output ae/results/my-quick
python ae/run.py --model ae/models/LLaDA2.0-mini --profile full \
  --rates 0.5 1 2 4 --output ae/results/my-full
```

Quick uses 10-second warmup and 30-second offered traffic per point, seed 42.
Full uses 120-second warmup and 600-second measurement, seeds 42/43/44; allow
several hours including queue drain. Warmups and measurements drain separately.
Full mode has a longer request timeout; request failures invalidate the run.

Paired methods receive the same fixed prompt order and seeded Poisson arrivals.
Performance requests generate exactly 128 tokens with EOS stopping disabled.
Both methods use TP=2, Triton attention, disabled radix cache, CUDA graphs and
`--mem-fraction-static 0.6`. Batch cap is 8 for `quick`, 64 for `sweep`; record
it with your results. PAS uses `ae/configs/pas-final.yaml` (horizon 8, confidence
floor 0.8, return ratio 0.75, stability gate on, aging 0.15). SGLang uses
threshold 0.95 and its baseline fallback. The paper's floor of 0.6 is kept as
`ae/configs/pas-paper.yaml` for direct comparison, but it is not the default: on
200 GSM8K examples it scores 155/200 against 182/200 for SGLang, whereas the
default point stays within about one percent (see `VALIDATION.md`).

Environment switches: `AE_CUDA_GRAPH=0` disables CUDA graphs, `AE_MEM_FRACTION`
changes the static memory fraction, `AE_PAS_CONFIG` selects a different PAS
config file. Exact commands/configurations are saved with each run. See
`CHANGES.md` for source provenance and the partial-block lifecycle correction.

Expected trend: P90 rises with offered load; PAS reduces latency and supports
higher throughput under load. With CUDA graphs on, the A100 reference run shows
PAS P90 1.25–2.9× lower than SGLang up to 8 req/s and 2.4–4.8× around the knee at
16–32 req/s, with peak throughput 7.15–15.52 req/s against 5.39–7.91. MT-Bench is
the exception at the two lowest load points, where PAS is 2–11% slower.

Every run must complete all offered requests; a failed request exits non-zero and
invalidates that run. Compare the whole curve rather than requiring an exact
speedup at every point, and repeat seeds when making statistical claims.

### Four-dataset load sweep

```bash
bash ae/reproduce.sh sweep
```

Runs SGLang and PAS sequentially over `gsm8k`, `mgsm`, `humaneval` and `mt_bench`
at 0.5, 1, 2, 4, 8, 16 and 32 req/s, then writes `summary.json/csv` and
`sweep.png/pdf`. Two server starts total. Add `--rates 64` at the end if the GPUs
have headroom; on the validated pair it measures queueing beyond saturation.

The HumanEval, MGSM and MT-Bench files are **load-sweep inputs only**. They are
not scored by this stage: `ae/scoring.py` implements exact match for GSM8K/MGSM
and code extraction for HumanEval, but the quality stage below reports GSM8K
only, and MT-Bench requires a judge model that is not shipped.

## Azure replay and length distribution

```bash
bash ae/reproduce.sh trace
```

`reproduce.sh trace` replays the first 300 Azure records, scales input/output
lengths by 1/32 and 1/4 (ceil, minimum one token), with caps 2048/256, and
uniformly scales arrival gaps to mean 20 req/s.

The rate matters more than any other setting here. At 2 req/s both systems are
arrival-limited on A100 and their throughput agrees within 1%; at 12 req/s PAS
shows a 2.25x P90 advantage but no throughput gain; from 20 req/s upward SGLang
saturates at about 14.5 req/s while PAS reaches about 23.5 req/s. The recorded
sweep (`ae/reference-results/azure-saturation/`) shows 2.25x/6.79x/3.31x/2.30x P90
at 12/20/30/45 req/s and a peak throughput gain of +62.6%. Always report the rate
with a trace result; a single low-rate point is not evidence about throughput. No Azure records in this prefix hit either cap; all transformations and
constructed token IDs are saved. This is a **scaled trace-derived workload**, not
an unchanged production workload.

Content comes from real GSM8K text repeated/truncated to the exact target token
count. It is a length surrogate, not the original Azure prompts. The client
checks served prompt/output lengths and rejects unresolved output masks.
Original relative timestamps and lengths can also be requested:

```bash
python ae/trace.py --model ae/models/LLaDA2.0-mini --mode original \
  --max-requests 0 --rate 0 --request-timeout 7200 --output ae/results/original-azure
```

Original mode filters input+output above the model's 32768-token context limit
and records the count. It can overload two A100s and take many hours; a complete
run of this mode has not been validated. Positive `--rate` uniformly rescales
gaps; always report the setting. `--max-requests` selects a prefix, not requests
chosen by observed performance.

Plots show completions within the measurement window and **client outstanding
requests**, not an estimated internal GPU queue. Drain is shown separately.
The length CDF uses the entire bundled window's original, unscaled token counts.

## Quality diagnostic

```bash
bash ae/reproduce.sh quality
# Larger diagnostic:
python ae/quality.py --model ae/models/LLaDA2.0-mini \
  --samples 64 --tokens 512 --output ae/results/my-quality
```

Uses natural EOS and numeric answer extraction (the `####` answer if present,
otherwise the last number) on GSM8K. Every response/reference and length
termination count is retained. The default sample count is 200
(`AE_QUALITY_SAMPLES`); the earlier 32-example check was too small to separate
the operating points and is no longer the default.

Reference measurement at the default operating point:

| Method | Correct | Exact match |
|---|---:|---:|
| SGLang | 183/200 | 0.915 |
| PAS (`pas-final.yaml`) | 177/200 | 0.885 |

Repeated 200-example runs gave PAS 177–183 and SGLang 182–183, so the gap is at
the edge of run-to-run variation. The artifact does not claim exact parity. With
CUDA graphs enabled, two runs of the same configuration can differ; treat a
single 200-example score as a range, not a constant (`VALIDATION.md`).

This is not HumanEval pass@1, MGSM exact match or MT-Bench judging.

## Outputs, metrics and validation

`ae/reference-results/` contains measured public reference records. Newly
measured output directories include commands, hardware/source/config identities,
warmups, request-level responses, errors, summaries and plots. Controlled
sweeps write `summary.json/csv` and `comparison.png/pdf`; trace runs also write
`azure-workload.json`, `traces.png/pdf` and unscaled length CDFs.

P90 is measured from scheduled arrival until response completion, including
client dispatch lag, HTTP, server queueing and execution. Drain latencies are
included. `measurement_completions_per_s` counts only window completions;
`drained_requests_per_s` divides all successful completions by measurement plus
drain. Plotted sweep throughput uses the latter. Failures remain recorded and
invalidate the run. Three-seed means are plotted without claiming confidence
intervals. Smoke timings establish functionality, not a particular speedup.

```bash
python ae/plot.py ae/results/my-quick
PYTHONPATH=python python -m pytest -q ae/tests
```

Six tests cover partial-block resumption/writeback, baseline FIFO isolation,
drain/failure accounting, deterministic arrival schedules and exact surrogate
lengths. `python ae/scoring.py` runs a scoring self-check with no network or GPU. Inspect server logs on startup failure. Increase `--start-timeout` for
cold compilation; for OOM verify both GPUs are free. Record any changed batch
cap for both methods. Do not upgrade individual locked dependencies.

## AE scope and submission

See `CLAIMS.md`, `VALIDATION.md`, `AE_FORM_GUIDE.md` and `SUBMISSION_DRAFT.md`.
The paper's exact H100 maxima, 8B/100B panels, component attribution,
quality-matched/four-task claims and confidential Kimi experiments are not
established by this public reproduction. The MT-Bench, HumanEval and MGSM files
are load-sweep inputs, not a four-task quality matrix. Upload the revised paper, this README (plain text/PDF), and a
frozen source archive or Git URL. Available badge requests additionally need a
permanent archival record; the development branch alone is not such a record.
