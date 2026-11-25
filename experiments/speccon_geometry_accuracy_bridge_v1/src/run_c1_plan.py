#!/usr/bin/env python3
"""Run the frozen six-run MRPC reference calibration bridge sequentially."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def record(item):
    with (ROOT/'run_manifest.jsonl').open('a') as f:
        f.write(json.dumps(item,ensure_ascii=False)+'\n')


def main():
    plan=json.loads((ROOT/'configs/c1_run_plan.json').read_text())
    for job in plan:
        output=Path(job['output'])
        metrics=output/'metrics.json'
        if metrics.exists():
            result=json.loads(metrics.read_text())
            if result.get('state')=='complete':
                record({'id':job['id'],'state':'already_complete','metrics':str(metrics)})
                continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Partial output exists; investigate before rerun: {output}')
        output.mkdir(parents=True,exist_ok=True)
        record({'id':job['id'],'state':'running','started_at':time.time(),
                'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'command':job['command']})
        with Path(job['log']).open('w') as log:
            result=subprocess.run(job['command'],cwd=job['cwd'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            record({'id':job['id'],'state':'failed','returncode':result.returncode,'log':job['log']})
            raise SystemExit(result.returncode)
        payload=json.loads(metrics.read_text())
        if payload.get('state')!='complete' or payload.get('adapter_parameters')!=755712:
            raise RuntimeError(f'Invalid result: {metrics}')
        record({'id':job['id'],'state':'complete','metrics':str(metrics),
                'final_accuracy_percent':payload['final_accuracy_percent'],
                'optimizer_steps':payload['optimizer_steps'],'completed_at':time.time()})
        print(job['id'],'complete',payload['final_accuracy_percent'],flush=True)


if __name__=='__main__':main()
