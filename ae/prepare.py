#!/usr/bin/env python3
"""Download public inputs and record exact identities. No model weights in Git."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

GSM_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("ae/data"))
    p.add_argument("--model", type=Path, default=Path("ae/models/LLaDA2.0-mini"))
    p.add_argument("--revision", default="main", help="HF revision; downloaded content must match model-files.json")
    p.add_argument("--local-model", action="store_true")
    a = p.parse_args()
    a.data_dir.mkdir(parents=True, exist_ok=True)
    if not a.local_model:
        from huggingface_hub import HfApi, snapshot_download
        a.revision = HfApi().model_info("inclusionAI/LLaDA2.0-mini", revision=a.revision).sha
        snapshot_download("inclusionAI/LLaDA2.0-mini", revision=a.revision, local_dir=a.model)
    idx = json.loads((a.model / "model.safetensors.index.json").read_text())
    files = sorted(set(idx["weight_map"].values()) | {
        "config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json",
        "configuration_llada2_moe.py", "modeling_llada2_moe.py", "special_tokens_map.json",
    })
    model_manifest = {name: {"bytes": (a.model / name).stat().st_size,
                             "sha256": sha256(a.model / name)} for name in files}
    expected = json.loads(Path(__file__).with_name("model-files.json").read_text())
    if model_manifest != expected:
        raise RuntimeError("Checkpoint differs from the evaluated files in ae/model-files.json. "
                           "Use the matching HF revision; do not compare different checkpoints.")
    source = Path(__file__).parent / "datasets/gsm8k.jsonl"
    expected_data = json.loads(source.with_name("manifest.json").read_text())
    if sha256(source) != expected_data["sha256"]:
        raise RuntimeError("Bundled GSM8K checksum mismatch")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    target = a.data_dir / "gsm8k.jsonl"
    target.write_bytes(source.read_bytes())
    manifest = {"model_repo": "inclusionAI/LLaDA2.0-mini", "model_revision": None if a.local_model else a.revision,
                "local_model": a.local_model, "model_files": model_manifest,
                "gsm8k_revision": GSM_REVISION, "gsm8k_sha256": sha256(target), "samples": len(rows)}
    (a.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(files)} model files; prepared {len(rows)} GSM8K examples.")


if __name__ == "__main__":
    main()
