#!/usr/bin/env python3
"""Run the frozen head-matched pairs, keeping each pair on one GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
ASSIGN={('cola',777):0,('rte',2026):0,('cola',2026):2,('mrpc',777):2,('mrpc',2026):3,('rte',777):3}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(x):
 with (ROOT/'run_manifest.jsonl').open('a') as f:f.write(json.dumps(x,sort_keys=True)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,2,3],required=True);a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.gpu)
 plan=json.loads((ROOT/'configs/run_plan.json').read_text())
 for j in plan:
  if ASSIGN[(j['task'],j['seed'])]!=a.gpu:continue
  output=Path(j['output']);metrics=output/'metrics.json'
  if metrics.exists() and json.loads(metrics.read_text()).get('state')=='complete':
   print(j['id'],'already complete',flush=True);continue
  if output.exists() and any(output.iterdir()):raise RuntimeError(f'Partial output: {output}')
  spec=Path(j['command'][j['command'].index('--extension-spec')+1])
  assert digest(spec)==j['spec_sha256']
  assert digest(json.loads(spec.read_text())['fixed_rank_pattern'])==j['rank_map_sha256']
  output.mkdir(parents=True,exist_ok=True)
  record({'id':j['id'],'state':'running','time':time.time(),'gpu':a.gpu,'command':j['command']})
  with Path(j['log']).open('w') as f:result=subprocess.run(j['command'],cwd=j['cwd'],stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:
   record({'id':j['id'],'state':'failed','time':time.time(),'returncode':result.returncode,'log':j['log']})
   raise SystemExit(result.returncode)
  m=json.loads(metrics.read_text());alloc=json.loads((output/'allocation.json').read_text())
  assert m['state']=='complete' and m['adapter_parameters']==755712
  assert alloc['rank_pattern']==json.loads(Path(json.loads(spec.read_text())['fixed_rank_pattern']).read_text())
  assert json.loads((output/'run_config.json').read_text())['model_init_seed']==j['seed']
  record({'id':j['id'],'state':'complete','time':time.time(),'metrics':str(metrics),'score':m['final_accuracy_percent']})
  print(j['id'],'complete',m['final_accuracy_percent'],flush=True)
if __name__=='__main__':main()
