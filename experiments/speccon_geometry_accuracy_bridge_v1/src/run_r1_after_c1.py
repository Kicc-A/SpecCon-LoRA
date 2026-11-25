#!/usr/bin/env python3
"""Start CoLA P1 only after the six preregistered C1 runs finish."""
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
    c1=json.loads((ROOT/'configs/c1_run_plan.json').read_text())
    plan=json.loads((ROOT/'configs/r1_run_plan.json').read_text())
    while True:
        finished=[]
        for job in c1:
            p=Path(job['output'])/'metrics.json'
            finished.append(p.exists() and json.loads(p.read_text()).get('state')=='complete')
        if all(finished):break
        if not subprocess.run(['tmux','has-session','-t','speccon_c1_20261108'],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            raise RuntimeError('C1 controller ended before all six results completed')
        time.sleep(30)
    print('C1 six runs complete; starting R1',flush=True)
    for job in plan:
        output=Path(job['output'])
        metrics=output/'metrics.json'
        if metrics.exists() and json.loads(metrics.read_text()).get('state')=='complete':
            record({'id':job['id'],'state':'already_complete','metrics':str(metrics)})
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Partial output exists: {output}')
        output.mkdir(parents=True,exist_ok=True)
        record({'id':job['id'],'state':'running','started_at':time.time(),
                'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'command':job['command']})
        with Path(job['log']).open('w') as log:
            result=subprocess.run(job['command'],cwd=job['cwd'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            record({'id':job['id'],'state':'failed','returncode':result.returncode,'log':job['log']})
            raise SystemExit(result.returncode)
        payload=json.loads(metrics.read_text())
        assert payload['state']=='complete' and payload['adapter_parameters']==755712
        record({'id':job['id'],'state':'complete','metrics':str(metrics),
                'final_mcc_percent':payload['final_score_percent'],'completed_at':time.time()})
        print(job['id'],'complete',payload['final_score_percent'],flush=True)


if __name__=='__main__':main()
