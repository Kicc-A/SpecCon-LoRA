import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bootstrap import digest
from launch import TRAIN, DEV, MODEL

TEST = '/workspace/SpecCon-LoRA/reproduction/gora_protocol_20261112/gsm8k_official_1319.jsonl'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--require-free-gpus', action='store_true')
    p.add_argument('--freeze-inputs', action='store_true', help='First setup only; never silently update hashes')
    a = p.parse_args()
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    for rel, expected in manifest['files'].items():
        assert digest(HERE / 'source' / rel) == expected, f'Source changed: {rel}'
    assert digest(HERE / 'evidence/gora_paper_v3.html') == manifest['paper_sha256']
    model = Path(MODEL)
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    paths = [Path(TRAIN), Path(DEV), Path(TEST)]
    paths += [model / n for n in ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json')]
    paths += [model / n for n in sorted(set(index['weight_map'].values()))]
    files = {str(f): {'sha256': digest(f), 'bytes': f.stat().st_size} for f in paths}
    rows = {}
    for file in (TRAIN, DEV, TEST):
        with open(file) as f:
            rows[file] = [json.loads(line) for line in f if line.strip()]
    assert len(rows[TRAIN]) == 100000
    assert len(rows[DEV]) == 10000
    assert len(rows[TEST]) == 1319
    assert all(isinstance(r.get('input'), str) and r.get('output') for r in rows[TRAIN])
    train_inputs = {r['input'].strip() for r in rows[TRAIN]}
    dev_inputs = {r['input'].strip() for r in rows[DEV]}
    # Exact string overlap is diagnostic only: MetaMath contains paraphrases/augmentations.
    train_dev_overlap = len(train_inputs & dev_inputs)
    input_lock = HERE / 'inputs.lock.json'
    if a.freeze_inputs:
        assert not input_lock.exists(), 'Input lock already exists; inspect, do not overwrite'
        input_lock.write_text(json.dumps(files, indent=2) + '\n')
    assert json.loads(input_lock.read_text()) == files, 'Dataset/model hash mismatch'
    packages = {}
    for name in ('torch', 'transformers', 'deepspeed', 'liger-kernel', 'numpy', 'triton', 'accelerate', 'flash-attn'):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    gpu_csv = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
    gpu_rows = [line.split(', ') for line in gpu_csv.strip().splitlines()]
    assert len(gpu_rows) >= 2
    if a.require_free_gpus:
        assert all(int(row[-1]) < 512 for row in gpu_rows[:2]), 'GPU 0/1 busy; refusing concurrent launch'
        assert shutil.which('ninja'), 'ninja is installed in the gora environment; put its bin directory on PATH'
        import torch
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam([torch.nn.Parameter(torch.zeros(2))], lr=5e-5, betas=(.9,.999), adamw_mode=True)
        del optimizer
    report = dict(status='PASS', python=sys.version, platform=platform.platform(), packages=packages,
                  source_files_verified=len(manifest['files']), input_files_verified=len(files),
                  rows={k: len(v) for k, v in rows.items()}, train_dev_exact_input_overlap=train_dev_overlap,
                  gpu_inventory=gpu_rows, global_batch='2 * 1 * 32 = 64',
                  limitations=['HF model entry, SDPA and activation checkpoint on 2x3090; upstream shell uses native llama and 8 GPUs',
                               'Exact author 100K sampling indices and original prompt JSON are not released/pinned; local subset is fingerprinted',
                               'No claim of bitwise or published-score reproduction'])
    target = HERE / 'preflight.latest.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
