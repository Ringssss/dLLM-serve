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
