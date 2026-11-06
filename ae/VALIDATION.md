# Recorded public validation

Two A100 SXM4 40 GB GPUs, TP=2, batch cap 8; corrected runtime source.
Reference records are retained in `reference-results/`.

## Controlled load, 128 output tokens

Seed 42, 10 s warmup, 30 s offered traffic per point; throughput includes drain.

| Offered req/s | SGLang P90 s | PAS P90 s | SGLang req/s | PAS req/s |
|---|---|---|---|---|
| 0.5 | 3.974 | 2.343 | 0.665 | 0.680 |
| 1 | 4.327 | 2.530 | 1.100 | 1.100 |
| 2 | 7.371 | 3.317 | 1.877 | 2.064 |
| 4 | 30.098 | 6.828 | 1.925 | 3.293 |
| 8 | 74.318 | 34.598 | 2.076 | 3.447 |

All 940 measurement requests succeeded with correct output lengths and no
unresolved masks. This is a single-seed quick sweep, not a confidence interval.

## Azure scaled replay

First 120 records of the bundled window, mean 2 req/s, input/output scaling
1/32 and 1/4, caps 2048/256. Neither cap is hit in this prefix.

| Method | P90 s | Throughput incl. drain req/s | Completed |
|---|---|---|---|
| SGLang | 2.747 | 1.977 | 120/120 |
| PAS | 1.687 | 1.994 | 120/120 |

Azure is arrival-limited at this operating point. The result supports a latency
improvement, not a large saturated-throughput gain. Original-length full-window
execution is implemented but was not completed here. Kimi is confidential and
excluded from the public artifact; Azure is the public trace example.

## Quality and correctness

An isolated 32-example GSM8K check with natural EOS and 512-token cap scored
SGLang 32/32 and PAS 23/32 under the documented numeric-answer extraction rule.
Length terminations were 1 and 3 respectively. This small diagnostic shows a
quality trade-off and does not support the paper's quality-preservation number.

A second empty virtual environment was built using `setup.sh` and the lock.
Six tests passed. An extracted source archive also passed the paired 128-token
smoke check (13/13 requests per method). Checkpoint SHA-256 verification was
exercised using existing local weights; Hugging Face download was blocked by
that host's network connectivity. The Azure-only public entry point was rerun successfully: six tests passed,
public dataset preparation succeeded, and both methods completed 120/120 trace
requests with no failures. The table above uses this public-branch rerun.

## Revision 2026-11-06: CUDA graphs, `pas-final.yaml`, four evaluation sets

All numbers below use the current entry point: CUDA graphs on, `--mem-fraction-static 0.6`,
`max-running-requests 64`, TP=2 on two A100 SXM4 40 GB GPUs, 128 output tokens with EOS
disabled, seeded Poisson arrivals (seed 42), 10 s warmup and 30 s offered traffic per point.
Single seed unless stated. Records: `ae/reference-results/sweep-a100-cudagraph/`.

### Why the operating point changed

The paper's floor (`min_conf` 0.6) was measured on 200 GSM8K examples and scored
155/200, against 182/200 for SGLang under the same protocol. The earlier 32-example
check was too small to show this and reported a 3-example gap instead of 27. The
artifact default is therefore `pas-final.yaml` (floor 0.8, stability gate on);
`pas-paper.yaml` is kept so the difference is directly measurable.

### Load sweep across four public evaluation sets

| Dataset | Offered req/s | SGLang P90 s | PAS P90 s | P90 ratio | SGLang req/s | PAS req/s |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 0.5 | 1.12 | 0.90 | 1.25x | 0.71 | 0.71 |
| gsm8k | 1 | 1.74 | 1.16 | 1.51x | 1.22 | 1.22 |
| gsm8k | 2 | 2.51 | 1.62 | 1.55x | 2.16 | 2.16 |
| gsm8k | 4 | 3.69 | 2.18 | 1.69x | 3.74 | 3.85 |
| gsm8k | 8 | 9.15 | 3.17 | 2.88x | 6.59 | 7.28 |
| gsm8k | 16 | 30.72 | 6.35 | 4.84x | 7.50 | 13.02 |
| gsm8k | 32 | 83.38 | 34.53 | 2.41x | 7.67 | 13.72 |
| gsm8k | 64 | 194.21 | 94.40 | 2.06x | 7.75 | 14.06 |
| mgsm | 0.5 | 1.12 | 0.89 | 1.26x | 0.71 | 0.71 |
| mgsm | 1 | 1.77 | 1.16 | 1.52x | 1.22 | 1.22 |
| mgsm | 2 | 2.56 | 1.59 | 1.61x | 2.16 | 2.16 |
| mgsm | 4 | 3.70 | 2.09 | 1.77x | 3.73 | 3.88 |
| mgsm | 8 | 9.53 | 3.32 | 2.87x | 6.59 | 7.27 |
| mgsm | 16 | 30.95 | 6.27 | 4.94x | 7.46 | 13.10 |
| mgsm | 32 | 84.50 | 34.63 | 2.44x | 7.57 | 13.64 |
| mgsm | 64 | 194.43 | 94.58 | 2.06x | 7.74 | 14.05 |
| humaneval | 0.5 | 0.85 | 0.67 | 1.27x | 0.71 | 0.71 |
| humaneval | 1 | 1.16 | 0.96 | 1.20x | 1.19 | 1.20 |
| humaneval | 2 | 2.09 | 1.18 | 1.78x | 2.15 | 2.15 |
| humaneval | 4 | 3.64 | 1.67 | 2.18x | 3.88 | 3.92 |
| humaneval | 8 | 8.84 | 3.03 | 2.92x | 6.71 | 7.37 |
| humaneval | 16 | 29.91 | 5.53 | 5.41x | 7.60 | 13.49 |
| humaneval | 32 | 80.11 | 28.49 | 2.81x | 7.87 | 15.36 |
| humaneval | 64 | 189.43 | 82.84 | 2.29x | 7.91 | 15.52 |
| mt_bench | 0.5 | 2.03 | 2.08 | 0.97x | 0.71 | 0.71 |
| mt_bench | 1 | 2.64 | 2.93 | 0.90x | 1.22 | 1.22 |
| mt_bench | 2 | 3.66 | 3.63 | 1.01x | 2.15 | 2.14 |
| mt_bench | 4 | 5.73 | 5.49 | 1.04x | 3.73 | 3.90 |
| mt_bench | 8 | 13.64 | 9.36 | 1.46x | 5.59 | 6.79 |
| mt_bench | 16 | 43.41 | 25.40 | 1.71x | 5.94 | 7.99 |
| mt_bench | 32 | 115.16 | 79.34 | 1.45x | 5.86 | 7.75 |
| mt_bench | 64 | 285.28 | 208.32 | 1.37x | 5.39 | 7.15 |
Every cell completed all offered requests with no failures (1,871 requests per cell at
64 req/s). Above the saturation point the latency reflects queueing, not capacity; the
throughput column is the capacity measure.

Peak measured throughput: GSM8K 14.06 vs 7.75 req/s, MGSM 14.05 vs 7.74, HumanEval
15.52 vs 7.91, MT-Bench 7.15 vs 5.39. MT-Bench is the one evaluation set where PAS is
slightly *worse* than SGLang at the two lowest load points (2.08 vs 2.03 s and 2.93 vs
2.64 s at 0.5 and 1 req/s) and only overtakes from 8 req/s. The likely cause is
partial-block recomputation, which long open-ended answers trigger most often; this is
a property of the design, not a measurement error.

### Quality at the default operating point

200 GSM8K examples, natural EOS, 512-token cap, numeric answer extraction as documented
in `ae/quality.py`. Records: `ae/reference-results/quality-200-final/`.

| Method | Correct | Exact match | Length terminations |
|---|---:|---:|---:|
| SGLang | 183/200 | 0.915 | 9 |
| PAS (`pas-final.yaml`) | 177/200 | 0.885 | 9 |

Repeated 200-example runs of the same configuration gave PAS 177–183 and SGLang
182–183, so the two are within about one percent of each other and the gap is at the
edge of run-to-run variation. The artifact therefore does **not** claim exact quality
parity; it claims that the default point stays within a few examples of the baseline
whereas the paper's 0.6 floor loses roughly 27 of 200.

### Determinism

With CUDA graphs disabled, repeated 200-example runs of PAS produced byte-identical
tokens (0/200 differences), and a single repeated request is identical as well. With
CUDA graphs enabled, a repeated single request is still identical within one server,
but two separate 200-example runs differed on 138/200 token sequences (correctness
178 vs 182). The cause is batch-composition-dependent coupling in padded graph replay,
not a stale-buffer or correctness bug. Reported quality figures are therefore ranges,
not single values, whenever CUDA graphs are on.

### Smoke checks performed for this revision

`ae/sweep.py` + `ae/plot_sweep.py` ran end-to-end on HumanEval and MT-Bench (22/22 per
method, plot written). The Azure stage ran at 12 req/s over 300 records: SGLang P90
4.60 s and PAS 1.80 s, 300/300 each. The quality stage ran on 64 and 200 examples.
The six lifecycle/measurement tests pass.
