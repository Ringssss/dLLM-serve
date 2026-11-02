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
