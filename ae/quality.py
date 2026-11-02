#!/usr/bin/env python3
"""GSM8K exact numeric-answer check, with natural EOS and saved generations."""
import argparse
import asyncio
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import aiohttp
from run import ROOT, server, prompt


def answer(text):
    match = re.search(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    if match is None:
        matches = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', text)
        token = matches[-1] if matches else None
    else:
        token = match.group(1)
    try:
        return str(Decimal(token.replace(',', '')).normalize()) if token else None
    except InvalidOperation:
        return None


async def evaluate(a, rows):
    records = []; sem = asyncio.Semaphore(4)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async def one(row):
            async with sem:
                body = {'text': prompt(row['question'] + '\nGive your final numeric answer after ####.'),
                        'sampling_params': {'temperature': 0, 'max_new_tokens': a.tokens}, 'stream': False}
                async with session.post(f'http://127.0.0.1:{a.port}/generate', json=body) as response:
                    data = await response.json()
                    if response.status != 200: raise RuntimeError(str(data))
                ids = data.get('output_ids')
                if not ids or 156895 in ids: raise RuntimeError('Missing tokens or unresolved output mask')
                pred, ref = answer(data['text']), answer(row['answer'])
                records.append({'id': row['id'], 'prediction': pred, 'reference': ref,
                                'correct': pred is not None and pred == ref, 'response': data})
        await asyncio.gather(*(one(row) for row in rows))
    return sorted(records, key=lambda r: r['id'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=64); p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--gpus', default='0,1'); p.add_argument('--tp', type=int, default=2)
    p.add_argument('--port', type=int, default=30174)
    a = p.parse_args(); a.batch = 8; a.start_timeout = 900
    if a.samples < 1 or a.tokens < 1 or len(a.gpus.split(',')) != a.tp: p.error('Invalid samples/tokens/GPU count')
    a.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (ROOT / 'ae/data/gsm8k.jsonl').read_text().splitlines()][:a.samples]
    summaries = []
    for method in ['SGLang', 'PAS']:
        print('Starting', method, flush=True)
        with server(a, method, a.output):
            records = asyncio.run(evaluate(a, rows))
        (a.output / f'{method}-generations.json').write_text(json.dumps(records, indent=2))
        summaries.append({'method': method, 'samples': len(records), 'correct': sum(r['correct'] for r in records),
                          'exact_match': sum(r['correct'] for r in records) / len(records),
                          'length_terminated': sum(r['response']['meta_info']['finish_reason']['type'] == 'length' for r in records)})
        (a.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
        print(summaries[-1], flush=True)
    (a.output / 'settings.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2))


if __name__ == '__main__': main()
