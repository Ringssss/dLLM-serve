# PAS artifact: public end-to-end reproduction

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
and installation. `all` runs the quick load sweep, Azure replay and a 32-example
GSM8K quality diagnostic. Expect about 15–30 minutes on the tested GPU pair,
plus initial downloads and compilation. Save `git rev-parse HEAD` with your run.
For a frozen AE submission, use its supplied commit or source archive.

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
Both methods use TP=2, batch cap 8, Triton attention, disabled radix cache and
CUDA graphs. PAS uses horizon 8, confidence floor .6, return ratio .75 and
imported aging coefficient .15. SGLang uses threshold .95 and its baseline
fallback. Exact commands/configurations are saved. See `CHANGES.md` for source
provenance and the partial-block lifecycle correction.

Expected trend: P90 rises with offered load; PAS reduces latency and supports
higher throughput under load. Low-load throughput can be arrival-limited.
Compare the whole curve rather than requiring an exact speedup at every point.
Retain unfavorable valid results and repeat seeds when making statistical claims.

## Azure replay and length distribution

```bash
bash ae/reproduce.sh trace
```

Quick replay takes the first 120 Azure records, scales input/output lengths by
1/32 and 1/4 (ceil, minimum one token), with caps 2048/256, and uniformly scales
arrival gaps to mean 2 req/s. No Azure records in this tested prefix hit either
cap. All transformations and constructed token IDs are saved. This is a
**scaled trace-derived workload**, not an unchanged production workload.

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
termination count is retained. The validated 32-example diagnostic scored
**SGLang 32/32 and PAS 23/32**. Thus the tested configuration exhibits a
performance–accuracy trade-off; it does not establish quality preservation.
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
lengths. Inspect server logs on startup failure. Increase `--start-timeout` for
cold compilation; for OOM verify both GPUs are free. Record any changed batch
cap for both methods. Do not upgrade individual locked dependencies.

## AE scope and submission

See `CLAIMS.md`, `VALIDATION.md` and `SUBMISSION_DRAFT.md`. The paper's exact
H100 maxima, 8B/100B panels, component attribution, quality-matched/four-task
claims and confidential Kimi experiments are not established by this public
reproduction. Upload the revised paper, this README (plain text/PDF), and a
frozen source archive or Git URL. Available badge requests additionally need a
permanent archival record; the development branch alone is not such a record.
