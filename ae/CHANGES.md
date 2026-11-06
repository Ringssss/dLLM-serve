# Changes from the original artifact

Original artifact commit: `8ab89eccb` (SGLang source plus historical scripts).
PAS source imported from `main` commit `1da50a96c` (V3 algorithm and request/scheduler mixins).

1. Added the missing `CW_SRPT_V3` implementation and its integration.
2. Corrected partial-block writeback: incomplete blocks remain internal state;
   only completed blocks append generated tokens. Resumed blocks retain positions,
   invalidate their mutable block KV and reuse the committed prefix.
3. Kept LowConfidence admission FIFO. The imported scheduler previously sorted
   decode requests regardless of which algorithm was selected.
4. Added an A100 capability guard to the shared MoE top-k route, selecting the
   existing fallback when FlashInfer's fused route does not support SM80.
5. Added a resolved environment lock, public-input preparation and checksums,
   sequential paired benchmark, correctness tests, raw-result plots and README.
6. Exposed the paper's operating point in YAML. Imported defaults (0.3 floor,
   0.5 return ratio) are overridden by 0.6 and 0.75 for these runs; aging remains 0.15.

This is an updated reproduction implementation. Historical JSON produced by
older code is not reclassified as evidence from this corrected implementation.
The full source of every run should be archived together with its results.

## Changes in this revision (2026-11-06)

7. **Operating point changed to `pas-final.yaml`.** The paper's floor
   (`min_conf` 0.6, `pas-paper.yaml`) loses measurable accuracy on 200 GSM8K
   examples, so the default entry point now uses 0.8 with the stability gate
   enabled; `pas-paper.yaml` is kept for direct comparison. See VALIDATION.md.
8. **CUDA graphs enabled by default.** Earlier runs used `--disable-cuda-graph`,
   which left each forward dominated by kernel-launch overhead. The entry point
   now keeps graphs on (`AE_CUDA_GRAPH=0` restores the old behaviour) and sets
   `--mem-fraction-static 0.6` (`AE_MEM_FRACTION`), because CUDA-graph capture
   and the fp32 logits buffer need headroom that the previous 0.8 setting did
   not leave on a 40 GB card.
9. **Algorithm work in `cw_srpt_v3.py`**: the trailing forward after the
   denoising loop reuses the last iteration's logits; confidence comes from
   `logsumexp` instead of a materialised `[batch, block, vocab]` softmax, and is
   computed in row chunks (`confidence_chunk_rows`) so memory stays bounded at
   high concurrency; the forced-progress fallback is branch-free so the host
   does not synchronise mid-loop; `stable_gate`, `saturate_threshold` and
   `confidence_chunk_rows` are new, default-off-with-opt-in config keys.
10. **Four public evaluation sets bundled** (`ae/datasets/*.jsonl` with
    `EVALSETS.json`, per-source licence files and SHA-256) and a new
    `ae/sweep.py` + `ae/plot_sweep.py` stage that measures the load sweep across
    all of them in one server start per method.
11. **Quality stage defaults to 200 GSM8K examples** (`AE_QUALITY_SAMPLES`),
    because the earlier 32-example check was too small to distinguish the
    operating points.
12. **Model acquisition falls back to the ModelScope mirror** when Hugging Face
    is unreachable; content is still verified against `ae/model-files.json`.

The reference recordings under `ae/reference-results/` were produced before
items 7-9 and are labelled as such; `sweep-a100-cudagraph/` is the current one.
