Reference sweep, 2x A100 SXM4 40 GB per method, TP=2, CUDA graphs on, max running requests 64, mem-fraction 0.6.
PAS config: ae/configs/pas-final.yaml (min_conf 0.8 + stability gate). 128 output tokens, EOS disabled, seeded Poisson arrivals (seed 42), 10 s warmup and 30 s offered traffic per point, single seed.
Produced by the development harness with the same request/arrival logic as ae/sweep.py; ae/sweep.py itself was smoke-tested separately. Regenerate with: bash ae/reproduce.sh sweep.
Throughput here is completions divided by wall time including drain. Rates above saturation measure queueing, not capacity.
