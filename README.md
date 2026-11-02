# PAS — PPoPP 2027 artifact

[Reproduction guide](ae/README.md) · [Measured results](ae/VALIDATION.md) · [Claim coverage](ae/CLAIMS.md)

This branch provides the serving source, locked environment, public evaluation
inputs and one-command comparisons of **SGLang LowConfidence and PAS** using
LLaDA2.0-mini. The public trace example is **Azure**. **Kimi traces are
confidential and cannot be released; Kimi reproduction is outside this public
artifact's scope.**

```bash
git clone --branch ppopp-artifacts https://github.com/Ringssss/dLLM-serve.git
cd dLLM-serve
bash ae/reproduce.sh all
```

The command creates a Python environment, downloads/verifies the public model,
prepares bundled GSM8K inputs, runs tests, and measures the paired load sweep,
Azure replay and a small quality check. Allow additional time and bandwidth for
the approximately 32.5 GB checkpoint. See the guide for hardware, quick smoke
testing, cached models and expected outputs.

The validated configuration uses two A100 40 GB GPUs. The recorded measurements
show improved performance and a quality trade-off at the tested operating point;
they are not a quality-matched acceleration claim. See `ae/VALIDATION.md`.

The runtime retains SGLang's Apache-2.0 license. Public datasets retain their
upstream licenses and attribution notices in `ae/datasets/` and `ae/traces/`.
