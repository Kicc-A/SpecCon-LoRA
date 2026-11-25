#!/usr/bin/env python3
"""Score all preregistered historical map variants on independent B1/B2."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'experiments/independent_calibration_mechanism_20261019_v1/source_snapshot'))
import torch
from speccon_lora.consensus import randomized_left_svd
from geometry_metrics import score_map


def load(task,arm):
    return torch.load(ROOT/'caches'/task/f'{arm}_raw.pt',map_location='cpu',weights_only=False)


def bases(payload):
    result={}
    for name,item in payload['gradients'].items():
        u,_=randomized_left_svd(item['grad'],16,oversample=8,power_iterations=1,seed=3407)
        result[name]=u
    return result


def main():
    maps=json.loads((ROOT/'data/rank_maps.json').read_text())
    fixed=json.loads((ROOT/'data/reference_A_fixed_rank_maps.json').read_text())
    rows=[]; details=[]; fixed_rows=[]
    for task in ('mrpc','cola','rte'):
        a=load(task,'A');au=bases(a)
        for probe in ('B1','B2'):
            b=load(task,probe)
            assert a['model_state_sha256']==b['model_state_sha256']
            bu=bases(b);grads={n:item['grad'] for n,item in b['gradients'].items()}
            for arm in ('F','E','S1','P1'):
                scored=score_map(fixed[task][arm],au,bu,grads)
                assert scored['adapter_budget']==755712
                fixed_rows.append({'task':task,'map':arm,'probe':probe,
                    'q_geom':scored['q_geom'],'q_grad':scored['q_grad'],
                    'adapter_budget':scored['adapter_budget'],
                    'accuracy_status':'missing_fixed_map_three_seed_training'})
            for seed in (42,777,2026):
                for arm in ('F','E','S1','P1'):
                    entry=maps[task][str(seed)][arm]
                    scored=score_map(entry['ranks'],au,bu,grads)
                    assert scored['adapter_budget']==755712
                    rows.append({'task':task,'seed':seed,'map':arm,'probe':probe,
                        'q_geom':scored['q_geom'],'q_grad':scored['q_grad'],
                        'adapter_budget':scored['adapter_budget'],
                        'rank_hash':entry['rank_hash'],'map_origin':entry['allocation'],
                        'accuracy_status':'missing' if arm=='P1' else 'historical_paired_only'})
                    for item in scored['details']:
                        details.append({'task':task,'seed':seed,'map':arm,'probe':probe,**item})
            print(task,probe,'scored',flush=True)
    for path,data in [(ROOT/'results/geometry_fixed_map_scores.csv',fixed_rows),
                      (ROOT/'results/geometry_map_scores.csv',rows),
                      (ROOT/'results/geometry_module_details.csv',details)]:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    print('complete',len(rows),'map-probe scores',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
