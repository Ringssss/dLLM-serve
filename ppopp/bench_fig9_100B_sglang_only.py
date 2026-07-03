#!/usr/bin/env python3
"""Run SGLang 100B trace replay only (Ours already done)."""
import csv, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np, requests as http_req

PYTHON='/home/zhujianian/miniconda3/envs/sglang-bench/bin/python'
MODEL='/mnt/models/LLaDA2.0-flash'; PORT=30100; TP=4; DUR=30; GEN=128
OUT=Path('/home/zhujianian/sglang/ppopp')

def load_prompts():
    p=[]
    for f in ['/home/zhujianian/morspec/data/gsm8k.jsonl','/home/zhujianian/morspec/data/humaneval.jsonl']:
        with open(f) as fh:
            for l in fh:
                d=json.loads(l); t=d.get('question',d.get('prompt',''))
                if t: p.append(t[:512])
    while len(p)<2000: p.extend(p[:2000-len(p)])
    return p[:2000]

def load_trace(path, dur_s, rate=3.0):
    ts=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            try: ts.append(datetime.fromisoformat(row['TIMESTAMP'].replace('+00:00','+00:00')).timestamp())
            except: continue
            if len(ts)>200000: break
    ts.sort()
    bs,bc,j=0,0,0
    for i in range(len(ts)):
        while j<len(ts) and ts[j]-ts[i]<=dur_s: j+=1
        if j-i>bc: bc=j-i; bs=i
    w=ts[bs:bs+bc]; t0=w[0]
    arr=[(t-t0,i%2000) for i,t in enumerate(w)]
    if arr:
        ar=len(arr)/(arr[-1][0]+1)
        if ar>0:
            s=ar/rate; arr=[(t*s,i) for t,i in arr]; arr=[(t,i) for t,i in arr if t<=dur_s]
    return arr

def mk(t): return '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+t+'<|role_end|><role>ASSISTANT</role>'

prompts=load_prompts()
print(f'Loaded {len(prompts)} prompts')

# Start SGLang server
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0,1,2,3'
for k in ['http_proxy','https_proxy','all_proxy','ALL_PROXY']: env.pop(k,None)
env['no_proxy']='127.0.0.1,localhost'
cmd=[PYTHON,'-m','sglang.launch_server','--model-path',MODEL,'--dllm-algorithm','LowConfidence',
     '--max-running-requests','8','--tp-size',str(TP),'--disable-radix-cache','--trust-remote-code','--port',str(PORT)]
proc=subprocess.Popen(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print('Starting SGLang 100B TP=4...')
for i in range(120):
    try:
        if http_req.get(f'http://127.0.0.1:{PORT}/health',timeout=5).status_code==200:
            print(f'Ready'); break
    except: pass
    time.sleep(5)

# Warmup
try: http_req.post(f'http://127.0.0.1:{PORT}/v1/completions',json={'model':'default','prompt':mk(prompts[0]),'max_tokens':32,'temperature':0},timeout=120)
except: pass

results={}
for tname,tpath in [('Kimi trace','/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv'),
                     ('Azure trace','/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv')]:
    print(f'\n=== SGLang | {tname} | {DUR} min ===')
    arr=load_trace(tpath,DUR*60,rate=3.0)
    print(f'  {len(arr)} arrivals')
    comp=[0]*DUR; ts=time.time()
    def send(p,st):
        now=time.time()-ts
        if st>now: time.sleep(st-now)
        try:
            r=http_req.post(f'http://127.0.0.1:{PORT}/v1/completions',json={'model':'default','prompt':mk(p),'max_tokens':GEN,'temperature':0},timeout=300)
            if r.status_code==200: comp[min(int((time.time()-ts)/60),DUR-1)]+=1
        except: pass
    with ThreadPoolExecutor(max_workers=64) as pool:
        futs=[]
        for at,pi in arr:
            if at>DUR*60: break
            futs.append(pool.submit(send,prompts[pi],at))
        for f in as_completed(futs,timeout=DUR*60+180): pass
    tp=[c/60.0 for c in comp]
    results[tname]=tp
    nz=[v for v in tp if v>0]
    print(f'  Mean: {np.mean(nz):.2f} req/s')

os.system(f'pkill -f "sglang.launch_server.*{PORT}"')

# Combine with Ours
with open(OUT/'fig9_100B_ours_data.json') as f:
    od=json.load(f)
combined={'time_min':list(range(DUR)),'data':{}}
for tn in results:
    combined['data'][tn]={'SGLang':results[tn],'Ours':od['data'][tn]['Ours']}
with open(OUT/'fig9_100B_plot_data.json','w') as f:
    json.dump(combined,f,indent=2)
print(f'\n✅ Saved: {OUT}/fig9_100B_plot_data.json')
for tn in combined['data']:
    sg=np.mean([v for v in combined['data'][tn]['SGLang'] if v>0])
    ou=np.mean([v for v in combined['data'][tn]['Ours'] if v>0])
    print(f'{tn}: SGLang={sg:.2f}, Ours={ou:.2f}, gain=+{(ou/sg-1)*100:.0f}%')
