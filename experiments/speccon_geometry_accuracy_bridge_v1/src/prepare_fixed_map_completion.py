#!/usr/bin/env python3
"""Freeze the outstanding fixed A_ref rank-map training matrix."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
MAPS = json.loads((ROOT/'data/reference_A_fixed_rank_maps.json').read_text())
SEEDS = (42,777,2026)
TASKS = {'cola': {'maps':'F E S1 P1','epochs':42,'lr':0.00019,'wd':0.196,'dropout':0.0,'rdrop':1.0,'label_smoothing':0.0,'lambda':1.5,'gpu':0},
         'rte': {'maps':'F E S1 P1','epochs':63,'lr':0.00021,'wd':0.198,'dropout':0.0,'rdrop':1.0,'label_smoothing':0.0,'lambda':0.5,'gpu':2},
         'mrpc': {'maps':'S1 P1','epochs':57,'lr':0.00035,'wd':0.307,'dropout':0.05,'rdrop':1.0,'label_smoothing':0.05,'lambda':1.5,'gpu':3}}
REF = ROOT/'runs/c1_mrpc_ref_lambda1.5_s42/allocation.json'
assert REF.exists()
plan=[]
for task, cfg in TASKS.items():
    raw=ROOT/f'caches/{task}/A_raw.pt'
    cache=torch.load(raw,map_location='cpu',weights_only=False)
    assert cache['ids']==list(range(512)) and cache['dropout_seed']==7001
    assert cache['model_state_sha256']=='2dc09c06fc39eadb48a86fc96f81b3b85b1cf69bc5804dcca1fa2b30f535bb97'
    for arm in cfg['maps'].split():
        ranks=MAPS[task][arm]
        assert len(ranks)==72 and set(ranks)==set(cache['gradients'])
        assert all(2<=r<=16 for r in ranks.values())
        budget=sum(r*sum(cache['gradients'][n]['grad'].shape) for n,r in ranks.items())
        assert budget==755712,(task,arm,budget)
        rank_path=ROOT/f'configs/fixed_{task}_{arm}_rank_pattern.json'
        rank_path.write_text(json.dumps(ranks,sort_keys=True,indent=2)+'\n')
        spec={'group':'G1_fixed_map','name':f'{task}_{arm}_Aref','task':task,'budget':755712,
              'allocation':'fixed','fixed_rank_pattern':str(rank_path),'init':'svd','steps':16,'blocks':4,
              'gamma':1.2,'lambda':0.0 if arm=='E' else cfg['lambda'],'two_forwards':True,
              'rdrop_alpha':cfg['rdrop'],'independent_raw':str(raw),'independent_ids':list(range(512)),
              'model_initialization_seed':42,'map_name':arm,'map_sha256':hashlib.sha256(rank_path.read_bytes()).hexdigest(),
              'expected_classifier_sha256':'8525992b7c0f62ea5e7d4d9ba33a20aa71acf73016426ef8315c03aa6386e112'}
        spec_path=ROOT/f'configs/fixed_{task}_{arm}.json'
        spec_path.write_text(json.dumps(spec,sort_keys=True,indent=2)+'\n')
        for seed in SEEDS:
            if task=='cola': rng=Path(f'/workspace/SpecCon-LoRA/experiments/glue_paper_extras_gpu3_20261016_v1/references/cola/seed{seed}/training_rng.pt')
            elif task=='rte': rng=Path(f'/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3/references/rte/seed{seed}/training_rng.pt')
            else: rng=Path(f'/workspace/SpecCon-LoRA/experiments/extended_ablation_20261021_v2/references/mrpc/seed{seed}/training_rng.pt')
            assert rng.exists(),rng
            run_id=f'fixed_{task}_{arm}_s{seed}'
            output=ROOT/'runs'/run_id
            cmd=['/root/anaconda3/bin/python','-u',str(ROOT/'fixed_map_source/reproduction/run_roberta_small_glue_regularization.py'),
                 '--task',task,'--seed',str(seed),'--output-dir',str(output),'--epochs',str(cfg['epochs']),
                 '--scheduler-epochs',str(cfg['epochs']),'--learning-rate',str(cfg['lr']),
                 '--weight-decay',str(cfg['wd']),'--lora-dropout',str(cfg['dropout']),'--lr-scheduler','linear',
                 '--calibration-steps','16','--rdrop-alpha',str(cfg['rdrop']),'--label-smoothing',str(cfg['label_smoothing']),
                 '--ablation-variant','F','--extension-spec',str(spec_path),
                 '--subspace-consistency-lambda',str(spec['lambda']),'--allocation-reference',str(REF),
                 '--training-rng-reference',str(rng)]
            plan.append({'id':run_id,'task':task,'map':arm,'seed':seed,'gpu':cfg['gpu'],'command':cmd,
                         'cwd':str(ROOT/'fixed_map_source'),'output':str(output),'log':str(ROOT/'logs'/f'{run_id}.log'),
                         'rank_map_sha256':spec['map_sha256'],'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                         'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'state':'planned'})
assert len(plan)==30
(ROOT/'configs/fixed_map_completion_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('frozen_jobs',len(plan),'by_task', {t:sum(j['task']==t for j in plan) for t in TASKS})
