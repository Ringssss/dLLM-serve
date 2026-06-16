#!/bin/bash
# bench.sh — Run the full A/B benchmark: LowConfidence vs CW-SRPT V2
# Usage: bash bench.sh [--tp 1] [--port 30100] [--model /path/to/model] [--reqs 64]
set -e

TP=1
PORT=30100
MODEL="/mnt/models/LLaDA2.0-mini"
N_REQS=64
RATES="5 10 20"

while [[ $# -gt 0 ]]; do
    case $1 in
        --tp) TP="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --reqs) N_REQS="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

GPUS=$(seq -s, 0 $((TP-1)))

send_requests() {
    local ALGO=$1 RATE=$2
    python3 -c "
import requests, time, json, numpy as np, threading

URL = 'http://localhost:${PORT}/v1/completions'
MODEL = '${MODEL}'
N = ${N_REQS}
RATE = ${RATE}

prompts = []
with open('benchmarks/humaneval_prompts.txt') as f:
    prompts = [l.strip() for l in f if l.strip()]
if not prompts:
    prompts = ['Write a Python function that checks if a number is prime.'] * N
while len(prompts) < N: prompts = prompts + prompts
prompts = prompts[:N]

def mk(t): return '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + t + '<|role_end|><role>ASSISTANT</role>'

results = []
np.random.seed(42)
delays = np.cumsum(np.random.exponential(1/RATE, N)).tolist()

def go(idx, delay):
    time.sleep(delay)
    t0 = time.perf_counter()
    try:
        r = requests.post(URL, json={'model': MODEL, 'prompt': mk(prompts[idx]), 'max_tokens': 128, 'temperature': 0}, timeout=120)
        lat = (time.perf_counter()-t0)*1000
        tok = r.json().get('usage',{}).get('completion_tokens',0)
        results.append((lat, tok))
    except Exception as e:
        results.append(((time.perf_counter()-t0)*1000, 0))

threads = [threading.Thread(target=go, args=(i, delays[i])) for i in range(N)]
t0 = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
wall = (time.perf_counter()-t0)*1000
lats = [r[0] for r in results]
toks = [r[1] for r in results]
tps = sum(toks)/(wall/1000) if wall > 0 else 0
print(f'{RATE},{tps:.0f},{np.median(lats):.0f},{np.percentile(lats,90):.0f},{np.percentile(lats,99):.0f}')
"
}

# Extract prompts for benchmark
if [ ! -f benchmarks/humaneval_prompts.txt ]; then
    python3 -c "
import json
with open('/home/zhujianian/morspec/data/humaneval.jsonl') as f:
    for line in f:
        print(json.loads(line)['prompt'].replace('\n', '\\\\n'))
" > benchmarks/humaneval_prompts.txt 2>/dev/null || echo "Write a Python function." > benchmarks/humaneval_prompts.txt
fi

echo "========================================================================"
echo "  dLLM-serve Benchmark: TP=${TP}, ${N_REQS} reqs, gen=128"
echo "========================================================================"
echo ""

for ALGO in LowConfidence CW_SRPT_V2; do
    echo "--- Starting ${ALGO} server (TP=${TP}) ---"
    CUDA_VISIBLE_DEVICES=${GPUS} python -m sglang.launch_server \
        --model-path ${MODEL} \
        --dllm-algorithm ${ALGO} \
        --max-running-requests 8 \
        --disable-radix-cache \
        --trust-remote-code \
        --port ${PORT} \
        --tp-size ${TP} &
    SERVER_PID=$!

    # Wait for server
    for i in $(seq 1 120); do
        curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1 && break
        sleep 2
    done

    if ! curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "ERROR: ${ALGO} server failed to start"
        kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
        continue
    fi

    # Warmup
    curl -s -X POST http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"Hi\", \"max_tokens\": 32}" > /dev/null

    echo "${ALGO} (TP=${TP}):"
    echo "  Rate,TPS,P50,P90,P99"
    for RATE in ${RATES}; do
        echo -n "  "
        send_requests ${ALGO} ${RATE}
    done
    echo ""

    # Kill server
    kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
    sleep 3
done

echo "Done."
