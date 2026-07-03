#!/usr/bin/env python3
"""
Fig. 13 Quality Benchmark: Measure output quality for SGLang vs Ours.

Metrics:
  - Readable rate (output > 10 chars)
  - GSM8K: numerical answer accuracy (extract last number, compare to ref)
  - HumanEval: output non-empty + syntactically valid Python
  - MGSM/MT-Bench: readable rate

Runs both LowConfidence and CW_SRPT_V3, compares outputs.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests as http_req

PYTHON = "/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL = "/mnt/models/LLaDA2.0-mini"
PORT = 30100
OUT_DIR = Path("/home/zhujianian/sglang/ppopp")

DATASETS = {
    "HumanEval": "/home/zhujianian/morspec/data/humaneval.jsonl",
    "GSM8K": "/home/zhujianian/morspec/data/gsm8k.jsonl",
    "MGSM": "/home/zhujianian/morspec/data/mgsm.jsonl",
    "MT-Bench": "/home/zhujianian/morspec/data/mt_bench.jsonl",
}
N_SAMPLES = 32  # per dataset


def mk_prompt(text):
    return ('<role>SYSTEM</role>detailed thinking off<|role_end|>'
            '<role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>')


def load_prompts(path, n):
    prompts = []
    refs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("question", d.get("prompt", ""))
            if not text:
                turns = d.get("turns", [])
                if turns:
                    text = turns[0]
            ref = d.get("answer", "")
            if text:
                prompts.append(text[:512])
                refs.append(ref)
            if len(prompts) >= n:
                break
    return prompts, refs


def wait_server(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = http_req.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    return False


def kill_server():
    os.system(f"pkill -f 'sglang.launch_server.*{PORT}' 2>/dev/null")
    time.sleep(3)


def start_server(algo):
    kill_server()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4"
    for k in ["http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)
    env["no_proxy"] = "127.0.0.1,localhost"
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--dllm-algorithm", algo,
           "--max-running-requests", "8",
           "--disable-radix-cache", "--trust-remote-code", "--port", str(PORT)]
    proc = subprocess.Popen(cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_server():
        proc.kill()
        return None
    return proc


def generate(prompt, max_tokens=128):
    try:
        r = http_req.post(f"http://127.0.0.1:{PORT}/v1/completions",
                          json={"model": "default", "prompt": mk_prompt(prompt),
                                "max_tokens": max_tokens, "temperature": 0}, timeout=120)
        if r.status_code == 200:
            return r.json()["choices"][0]["text"]
    except:
        pass
    return ""


def extract_number(text):
    """Extract last number from text (for GSM8K accuracy)."""
    numbers = re.findall(r'[-+]?\d*\.?\d+', text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def gsm8k_accuracy(outputs, refs):
    """Check if extracted numerical answer matches reference."""
    correct = 0
    total = 0
    for out, ref in zip(outputs, refs):
        pred = extract_number(out)
        # Extract answer from ref (format: "#### 42")
        ref_match = re.search(r'####\s*([-+]?\d*\.?\d+)', ref.replace(",", ""))
        if ref_match:
            ref_num = float(ref_match.group(1))
            total += 1
            if pred is not None and abs(pred - ref_num) < 0.01:
                correct += 1
    return correct, total


def main():
    algos = ["LowConfidence", "CW_SRPT_V3"]
    all_results = {}

    for algo in algos:
        print(f"\n{'='*50}")
        print(f"Algorithm: {algo}")
        print(f"{'='*50}")

        proc = start_server(algo)
        if proc is None:
            print("  FAILED to start")
            continue

        algo_results = {}
        for ds_name, ds_path in DATASETS.items():
            prompts, refs = load_prompts(ds_path, N_SAMPLES)
            print(f"  {ds_name} ({len(prompts)} samples)...", end=" ", flush=True)

            outputs = []
            for p in prompts:
                out = generate(p)
                outputs.append(out)

            # Readable rate
            readable = sum(1 for o in outputs if len(o.strip()) > 10)
            readable_rate = readable / len(outputs) * 100

            # Dataset-specific metrics
            extra = {}
            if ds_name == "GSM8K" and refs:
                correct, total = gsm8k_accuracy(outputs, refs)
                extra["accuracy"] = round(correct / max(total, 1) * 100, 1)
                extra["correct"] = correct
                extra["total"] = total
                print(f"readable={readable}/{len(outputs)}, acc={extra['accuracy']}%")
            elif ds_name == "HumanEval":
                # Check if output looks like valid Python
                valid_py = sum(1 for o in outputs if "def " in o or "return" in o or "=" in o)
                extra["valid_python_rate"] = round(valid_py / len(outputs) * 100, 1)
                print(f"readable={readable}/{len(outputs)}, valid_py={valid_py}/{len(outputs)}")
            else:
                print(f"readable={readable}/{len(outputs)}")

            algo_results[ds_name] = {
                "readable": readable,
                "n": len(outputs),
                "readable_rate": round(readable_rate, 1),
                "samples": outputs[:2],
                **extra,
            }

        all_results[algo] = algo_results
        kill_server()

    # Save
    out_path = OUT_DIR / "fig13_quality_data.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {out_path}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Dataset':<12} {'Metric':<15} {'SGLang':<10} {'Ours':<10}")
    print("-" * 47)
    for ds in DATASETS:
        lc = all_results.get("LowConfidence", {}).get(ds, {})
        v3 = all_results.get("CW_SRPT_V3", {}).get(ds, {})
        print(f"{ds:<12} {'Readable %':<15} {lc.get('readable_rate', 'N/A'):<10} {v3.get('readable_rate', 'N/A'):<10}")
        if "accuracy" in lc:
            print(f"{'':12} {'GSM8K Acc %':<15} {lc['accuracy']:<10} {v3.get('accuracy', 'N/A'):<10}")
        if "valid_python_rate" in lc:
            print(f"{'':12} {'Valid Py %':<15} {lc['valid_python_rate']:<10} {v3.get('valid_python_rate', 'N/A'):<10}")


if __name__ == "__main__":
    main()
