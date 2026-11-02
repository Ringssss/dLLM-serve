#!/usr/bin/env python3
"""Freeze tracked source plus AE files; omit models, caches and transient runs."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parents[1]
names = set(subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines())
names.update(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True).splitlines())
files = sorted(n for n in names if (root / n).is_file() or (root / n).is_symlink())
manifest = {}
for name in files:
    p = root / name
    if not p.is_symlink(): manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
manifest.pop('ae/SHA256SUMS.json', None)
(root / 'ae/SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n')
files = sorted(set(files) | {'ae/SHA256SUMS.json'})
archive = root.parent / 'PAS-PPoPP27-AE-public.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for name in files:
        tar.add(root / name, arcname='PAS-PPoPP27-AE/' + name, recursive=False)
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(archive.suffix + '.sha256').write_text(checksum + '  ' + archive.name + '\n')
print(archive, archive.stat().st_size, checksum)
