#!/usr/bin/env python3
"""Fig 9 100B v2: target_rate=8 to stress SGLang. Run both in parallel on GPU 0-3 and 4-7."""
import csv,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path
import numpy as np,requests as http_req

PYTHON='/home/zhujianian/miniconda3/envs/sglang-bench/bin/python'
MODEL='/mnt/models/LLaDA2.0-flash';TP=4;DUR=30;GEN=128
OUT=Path('/home/zhujianian/sglang/ppopp')
TARGET_RATE=8.0

def load_prompts():
    p=[]
    for f in ['/home/zhujianian/morspec/data/gsm8k.jsonl','/home/zhujianian/morspec/data/humaneval.jsonl']:
        with open(f) as fh:
            for l in fh:
                d=json.loads(l);t=d.get('question',d.get('prompt',''))
                if t:p.append(t[:512])
    while len(p)<2000:p.extend(p[:2000-len(p)])
    return p[:2000]

def load_trace(path,dur_s):
    ts=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            try:ts.append(datetime.fromisoformat(row['TIMESTAMP'].replace('+00:00','+00:00')).timestamp())
            except:continue
            if len(ts)>200000:break
    ts.sort()
    bs,bc,j=0,0,0
    for i in range(len(ts)):
        while j<len(ts) and ts[j]-ts[i]<=dur_s:j+=1
        if j-i>bc:bc=j-i;bs=i
    w=ts[bs:bs+bc];t0=w[0]
    arr=[(t-t0,i%2000) for i,t in enumerate(w)]
    if arr:
        ar=len(arr)/(arr[-1][0]+1)
        if ar>0:s=ar/TARGET_RATE;arr=[(t*s,i) for t,i in arr];arr=[(t,i) for t,i in arr if t<=dur_s]
    return arr

def mk(t):return '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+t+'<|role_end|><role>ASSISTANT</role>'

def run_trace_experiment(algo, port, gpus, prompts, trace_name, trace_path):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpus
    for k in ['http_proxy','https_proxy','all_proxy','ALL_PROXY']:env.pop(k,None)
    env['no_proxy']='127.0.0.1,localhost'
    cmd=[PYTHON,'-m','sglang.launch_server','--model-path',MODEL,'--dllm-algorithm',algo,
         '--max-running-requests','8','--tp-size',str(TP),'--disable-radix-cache',
         '--trust-remote-code','--port',str(port)]
    proc=subprocess.Popen(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    method='Ours' if 'V3' in algo else 'SGLang'
    print(f'  [{method}] Starting on GPU {gpus}...')
    for i in range(120):
        try:
            if http_req.get(f'http://127.0.0.1:{port}/health',timeout=5).status_code==200:break
        except:pass
        time.sleep(5)
    try:http_req.post(f'http://127.0.0.1:{port}/v1/completions',json={'model':'default','prompt':mk(prompts[0]),'max_tokens':32,'temperature':0},timeout=120)
    except:pass
    print(f'  [{method}] Ready. Running {trace_name} for {DUR} min...')

    arr=load_trace(trace_path,DUR*60)
    comp=[0]*DUR;ts_start=time.time()
    def send(p,st):
        now=time.time()-ts_start
        if st>now:time.sleep(st-now)
        try:
            r=http_req.post(f'http://127.0.0.1:{port}/v1/completions',json={'model':'default','prompt':mk(p),'max_tokens':GEN,'temperature':0},timeout=300)
            if r.status_code==200:comp[min(int((time.time()-ts_start)/60),DUR-1)]+=1
        except:pass
    with ThreadPoolExecutor(max_workers=64) as pool:
        futs=[]
        for at,pi in arr:
            if at>DUR*60:break
            futs.append(pool.submit(send,prompts[pi],at))
        for f in as_completed(futs,timeout=DUR*60+180):pass
    tp=[c/60.0 for c in comp]
    os.system(f'pkill -f "sglang.launch_server.*{port}"');time.sleep(3)
    nz=[v for v in tp if v>0]
    print(f'  [{method}] {trace_name} done. Mean={np.mean(nz):.2f} Peak={max(tp):.2f}')
    return method, trace_name, tp

prompts=load_prompts()
print(f'Loaded {len(prompts)} prompts, target_rate={TARGET_RATE} req/s')

# Run SGLang and Ours in parallel on different GPUs, Kimi trace first
all_results={}
KIMI='/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv'
AZURE='/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv'

for trace_name,trace_path in [('Kimi trace',KIMI),('Azure trace',AZURE)]:
    print(f'\n{"="*50}\n{trace_name} @ {TARGET_RATE} req/s\n{"="*50}')
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1=pool.submit(run_trace_experiment,'LowConfidence',30100,'0,1,2,3',prompts,trace_name,trace_path)
        f2=pool.submit(run_trace_experiment,'CW_SRPT_V3',30200,'4,5,6,7',prompts,trace_name,trace_path)
        for f in [f1,f2]:
            method,tn,tp=f.result()
            all_results.setdefault(tn,{})[method]=tp

with open(OUT/'fig9_100B_plot_data.json','w') as f:
    json.dump({'time_min':list(range(DUR)),'data':all_results},f,indent=2)
print(f'\n✅ Saved: {OUT}/fig9_100B_plot_data.json')
for tn in all_results:
    sg=np.mean([v for v in all_results[tn].get('SGLang',[]) if v>0])
    ou=np.mean([v for v in all_results[tn].get('Ours',[]) if v>0])
    gain=(ou/sg-1)*100 if sg>0 else 0
    print(f'{tn}: SGLang={sg:.2f}, Ours={ou:.2f}, gain=+{gain:.0f}%')
