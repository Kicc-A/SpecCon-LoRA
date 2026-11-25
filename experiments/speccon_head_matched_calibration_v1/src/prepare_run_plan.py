#!/usr/bin/env python3
"""Freeze the 12 head-matched Full/Energy training commands."""
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/SpecCon-LoRA/experiments/speccon_geometry_accuracy_bridge_v1')
TASKS={'cola':dict(epochs=42,lr=0.00019,wd=0.196,dropout=0.0,label=0.0,lam=1.5),
       'mrpc':dict(epochs=57,lr=0.00035,wd=0.307,dropout=0.05,label=0.05,lam=1.5),
       'rte':dict(epochs=63,lr=0.00021,wd=0.198,dropout=0.0,label=0.0,lam=0.5)}
REF=BASE/'runs/c1_mrpc_ref_lambda1.5_s42/allocation.json'
assert REF.exists()
plan=[]
for task,cfg in TASKS.items():
 for seed in (777,2026):
  raw=ROOT/'caches'/f'{task}_head{seed}_A_raw.pt'
  data=torch.load(raw,map_location='cpu',weights_only=False)
  assert data['ids']==list(range(512)) and data['model_init_seed']==seed and data['dropout_seed']==7001
  if task=='cola': rng=Path(f'/workspace/SpecCon-LoRA/experiments/glue_paper_extras_gpu3_20261016_v1/references/cola/seed{seed}/training_rng.pt')
  elif task=='rte': rng=Path(f'/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3/references/rte/seed{seed}/training_rng.pt')
  else: rng=Path(f'/workspace/SpecCon-LoRA/experiments/extended_ablation_20261021_v2/references/mrpc/seed{seed}/training_rng.pt')
  assert rng.exists()
  for arm in ('F','E'):
   rank_path=ROOT/'rank_maps'/f'{task}_{arm}_head{seed}.json'
   ranks=json.loads(rank_path.read_text())
   assert len(ranks)==72 and all(2<=v<=16 for v in ranks.values())
   assert sum(v*sum(data['gradients'][n]['grad'].shape) for n,v in ranks.items())==755712
   spec=dict(group='head_matched_calibration',name=f'{task}_{arm}_head{seed}',task=task,seed=seed,
             budget=755712,allocation='fixed',fixed_rank_pattern=str(rank_path),init='svd',steps=16,blocks=4,
             gamma=1.2,**{'lambda':0.0 if arm=='E' else cfg['lam']},two_forwards=True,rdrop_alpha=1.0,
             independent_raw=str(raw),independent_ids=list(range(512)),model_initialization_seed=seed,
             expected_classifier_sha256=data['classifier_sha256'],
             map_sha256=hashlib.sha256(rank_path.read_bytes()).hexdigest())
   spec_path=ROOT/'configs'/f'{task}_{arm}_s{seed}.json'
   spec_path.write_text(json.dumps(spec,sort_keys=True,indent=2)+'\n')
   run_id=f'{task}_{arm}_s{seed}';out=ROOT/'runs'/run_id
   cmd=['/root/anaconda3/bin/python','-u',str(ROOT/'source_snapshot/reproduction/run_roberta_small_glue_regularization.py'),
        '--task',task,'--seed',str(seed),'--model-init-seed',str(seed),'--output-dir',str(out),
        '--epochs',str(cfg['epochs']),'--scheduler-epochs',str(cfg['epochs']),'--learning-rate',str(cfg['lr']),
        '--weight-decay',str(cfg['wd']),'--lora-dropout',str(cfg['dropout']),'--lr-scheduler','linear',
        '--calibration-steps','16','--rdrop-alpha','1.0','--label-smoothing',str(cfg['label']),
        '--ablation-variant','F','--extension-spec',str(spec_path),
        '--subspace-consistency-lambda',str(spec['lambda']),'--allocation-reference',str(REF),
        '--training-rng-reference',str(rng)]
   plan.append(dict(id=run_id,task=task,arm=arm,seed=seed,command=cmd,cwd=str(ROOT/'source_snapshot'),
                    output=str(out),log=str(ROOT/'logs'/f'{run_id}.log'),
                    rank_map_sha256=spec['map_sha256'],spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                    cache_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),state='planned'))
assert len(plan)==12
(ROOT/'configs/run_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('frozen',len(plan),'training runs')
