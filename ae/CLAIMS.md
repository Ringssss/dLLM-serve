# Public reproduction scope

| Paper item | Public artifact coverage |
|---|---|
| Figures 1–3, 5–7 | Conceptual illustrations, not numerical reproduction |
| Figure 4 / progress-lane attribution | No complete per-lane counting reproduction provided |
| Figure 8 | Unscaled Azure window length CDF provided |
| Figure 9 | Azure-derived completion and client-outstanding plots; not internal queue depth |
| Figure 10 / hindsight signal fidelity | Not reproduced |
| Figure 11 / Section 8.3 | Fresh paired mini-model controlled-load sweep |
| Figure 12 | Fresh scaled Azure replay; unscaled production equivalence and 100B excluded |
| Figure 13 / component attribution | Two-method comparison does not establish individual contributions |
| Section 8.5 / per-forward efficiency | Not reproduced |
| Figure 14 / quality | GSM8K diagnostic only; four-task quality preservation not established |
| Parameter robustness | Fixed explicit operating point, not a robustness claim |
| Kimi experiments | Confidential trace cannot be released; public reproduction uses Azure |

Only SGLang LowConfidence and PAS are executed. Exact H100 timings, additional
models and other baselines are outside the tested scope. Historical scripts
with local paths or constructed plot series have been removed from the branch
tip; the supported reproduction entry point is `ae/reproduce.sh`.
