#!/usr/bin/env python3
"""Extract fixed windows to anonymous arrival/length triples (never copy raw logs)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

FIELDS = ['arrival_s', 'input_tokens', 'output_tokens']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(part)
    return h.hexdigest()


def stamp(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--azure', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'traces')
    p.add_argument('--azure-start', default='2024-05-12T00:00:00+00:00')
    p.add_argument('--duration', type=int, default=1800)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    manifest = {'duration_s': a.duration, 'fields': FIELDS, 'sources': {}, 'windows': {}}
    az = []; begin = stamp(a.azure_start); invalid = 0
    with a.azure.open() as f:
        for row in csv.DictReader(f):
            t = stamp(row['TIMESTAMP'])
            if begin <= t < begin + a.duration:
                ni, no = int(row['ContextTokens']), int(row['GeneratedTokens'])
                if ni <= 0 or no <= 0: invalid += 1; continue
                az.append((t - begin, ni, no))
    manifest['sources']['azure'] = {'file': a.azure.name, 'sha256': sha(a.azure),
        'url': 'https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv'}
    manifest['windows']['azure'] = {'start_utc': a.azure_start, 'excluded_nonpositive_lengths': invalid}
    for name, rows in [('azure', az)]:
        if not rows: raise RuntimeError(f'No valid records for {name} window')
        rows.sort()
        path = a.output / f'{name}.csv'
        with path.open('w') as f:
            w = csv.writer(f); w.writerow(FIELDS); w.writerows(rows)
        manifest['windows'][name].update(records=len(rows), sha256=sha(path))
        print(name, len(rows), 'records', flush=True)
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__': main()
