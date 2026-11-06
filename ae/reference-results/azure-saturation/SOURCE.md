Azure trace saturation sweep, 2x A100 SXM4 40 GB per method, TP=2, CUDA graphs on, batch cap 64.
Same scaled workload as `reproduce.sh trace` (first 1000 records, input x1/32 cap 2048, output x1/4 cap 256;
0 input-cap hits, 2 output-cap hits), replayed at four target rates. 128 output tokens, EOS disabled.
PAS config ae/configs/pas-final.yaml. Methods ran on separate GPU pairs concurrently.
Per-request JSON is not archived (about 11 MB); regenerate with the sweep driver in the analysis notes or by
running `ae/trace.py` once per rate. Trace checksum: a08534828b9ece0e698187d2e3050573ef5d8bfc7459c575db9b4906e3785192.
