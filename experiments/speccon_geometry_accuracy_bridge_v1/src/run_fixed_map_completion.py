#!/usr/bin/env python3
"""Run one third of the frozen fixed-map matrix on a dedicated GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
GPUS=(0,2,3)

def append(row):
    with (ROOT/'run_manifest.jsonl').open('a') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,choices=range(3),required=True)
    args=parser.parse_args()
    gpu=GPUS[args.worker]
    jobs=json.loads((ROOT/'configs/fixed_map_completion_plan.json').read_text())
    jobs=[j for i,j in enumerate(jobs) if i%3==args.worker]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):
        raise RuntimeError(f'Worker {args.worker} requires only GPU {gpu} visible')
    for job in jobs:
        output=Path(job['output']); metrics=output/'metrics.json'
        if metrics.exists():
            result=json.loads(metrics.read_text())
            if result.get('state')=='complete' and result.get('adapter_parameters')==755712:
                print(job['id'],'already complete',flush=True)
                continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Partial run exists, inspect before resume: {output}')
        spec=Path(job['command'][job['command'].index('--extension-spec')+1])
        assert sha(spec)==job['spec_sha256']
        assert sha(json.loads(spec.read_text())['fixed_rank_pattern'])==job['rank_map_sha256']
        output.mkdir(parents=True,exist_ok=True)
        append({'id':job['id'],'stage':'G1_fixed_map','state':'running','time':time.time(),
                'gpu':gpu,'command':job['command'],'spec_sha256':job['spec_sha256'],
                'rank_map_sha256':job['rank_map_sha256']})
        with Path(job['log']).open('w') as log:
            result=subprocess.run(job['command'],cwd=job['cwd'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            append({'id':job['id'],'state':'failed','time':time.time(),'returncode':result.returncode,'log':job['log']})
            raise SystemExit(result.returncode)
        payload=json.loads(metrics.read_text()); alloc=json.loads((output/'allocation.json').read_text())
        expected=json.loads(Path(json.loads(spec.read_text())['fixed_rank_pattern']).read_text())
        if payload.get('state')!='complete' or payload.get('adapter_parameters')!=755712 or alloc['rank_pattern']!=expected:
            raise RuntimeError(f'Invalid result: {metrics}')
        append({'id':job['id'],'state':'complete','time':time.time(),'metrics':str(metrics),
                'score':payload['final_accuracy_percent'],'optimizer_steps':payload['optimizer_steps']})
        print(job['id'],'complete',payload['final_accuracy_percent'],flush=True)

if __name__=='__main__': main()
