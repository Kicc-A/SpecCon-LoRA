#!/usr/bin/env python3
"""Run three preregistered MRPC ensemble-map trainings sequentially."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def record(item):
    with (ROOT/'run_manifest.jsonl').open('a') as f:
        f.write(json.dumps(item)+'\n')


def main():
    jobs=json.loads((ROOT/'configs/run_plan.json').read_text())
    for job in jobs:
        output=Path(job['output']);metrics=output/'metrics.json'
        if metrics.exists() and json.loads(metrics.read_text()).get('state')=='complete':
            record({'id':job['id'],'state':'already_complete','metrics':str(metrics)})
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Nonempty incomplete output: {output}')
        output.mkdir(parents=True,exist_ok=True)
        record({'id':job['id'],'state':'running','started_at':time.time(),
                'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
        with Path(job['log']).open('w') as f:
            result=subprocess.run(job['command'],cwd=job['cwd'],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            record({'id':job['id'],'state':'failed','returncode':result.returncode,
                    'log':job['log']})
            raise SystemExit(result.returncode)
        payload=json.loads(metrics.read_text())
        assert payload['state']=='complete' and payload['adapter_parameters']==755712
        record({'id':job['id'],'state':'complete','metrics':str(metrics),
                'final_accuracy_percent':payload['final_accuracy_percent'],
                'final_step':payload['optimizer_steps'],'completed_at':time.time()})
        print(job['id'],payload['final_accuracy_percent'],flush=True)


if __name__=='__main__':main()
