# Public Azure trace window

`azure.csv` contains relative arrival seconds, input tokens and output tokens
for 44,102 requests in 2024-05-12 00:00–00:30 UTC. Extraction rules are in
`../prepare_traces.py`; source and subset SHA-256 values are in `manifest.json`.
Kimi traces are confidential and cannot be distributed. The public workflow
uses Azure as its example and has no Kimi input dependency.

Source: [Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md).
Attribution: Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Josep Torrellas, Esha
Choukse. *DynamoLLM: Designing LLM Inference Clusters for Performance and Energy
Efficiency*. HPCA 2025. License: CC-BY 4.0; see `AZURE-LICENSE.txt`.

Changes: select the fixed window, convert absolute timestamps to relative
seconds, and retain/rename the two token-count fields. Runtime scaling is
separately documented in the reproduction guide. Text content is a GSM8K
length surrogate, not original Azure prompts.
