# PPoPP'27 AE form draft

**Title:** PAS: An Efficient Block-Wise Scheduling Framework for Diffusion LLM Serving

**Artifact abstract:** The artifact provides PAS and a SGLang LowConfidence
baseline on the same LLaDA2.0-mini BF16 model, locked dependencies, public
GSM8K and Azure inputs, lifecycle tests, paired open-loop measurements and
plots generated from saved requests. It supports controlled-load and scaled
Azure trace-derived performance trends. Kimi traces are confidential and
cannot be released; Azure is the public reproduction example.

**Artifacts:** Upload the revised paper and `ae/README.md` as plain text/PDF.
Use a frozen commit or archive from
https://github.com/Ringssss/dLLM-serve/tree/ppopp-artifacts . Record the exact
commit used in the submission. Weights are downloaded from the public model
repository; public evaluation datasets are bundled.

**Hardware/software:** Linux x86-64, Python 3.12, NVIDIA driver supporting
CUDA 13. Validated on 2×A100 SXM4 40 GB, TP=2. Provision approximately 128 GB
host RAM and 120 GB disk, including the 32.5 GB model and dependencies.
No private machine, credentials or paid judge service is required.

**Runtime/scope:** Budget about 30 minutes for the getting-started test after
installation/download; the quick suite usually takes 15–30 minutes on the
tested pair. The full three-seed load sweep takes several hours including
drain. Reproduce the direction of mini-model latency/throughput trends and
scaled Azure replay, not exact H100 maxima. Additional models, other baselines,
component attribution, four-task quality preservation and Kimi are outside the
public scope. The recorded GSM8K diagnostic shows a quality trade-off.

**Badges:** Consider Functional and Results Reproduced for the agreed limited
scope. Available additionally requires a permanent public archival record;
enter a DOI only after one has actually been created. Functional/Reusable
are alternative levels.

**Authors/contacts/conflicts:** Enter confirmed ACM author names, affiliations,
corresponding author, contact addresses and conflicts. No form submission or
archival DOI creation has been performed by these scripts.
