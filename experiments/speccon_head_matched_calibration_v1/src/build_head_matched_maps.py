#!/usr/bin/env python3
"""Allocate frozen F and E rank maps from the captured per-head A_ref gradients."""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/SpecCon-LoRA/experiments/speccon_geometry_accuracy_bridge_v1')
sys.path.insert(0,str(ROOT/'source_snapshot'))
from speccon_lora.config import SpecConLoraConfig
from speccon_lora.consensus import build_consensus_spectral_profiles
from speccon_lora.rank_allocator import allocate_ranks

def main():
 old=json.loads((BASE/'data/reference_A_fixed_rank_maps.json').read_text())
 for task in ('cola','mrpc','rte'):
  for seed in (42,777,2026):
   path=(BASE/'caches'/task/'A_raw.pt') if seed==42 else ROOT/'caches'/f'{task}_head{seed}_A_raw.pt'
   if not path.exists(): raise FileNotFoundError(path)
   data=torch.load(path,map_location='cpu',weights_only=False)
   cfg=SpecConLoraConfig(**data['config'])
   profile=build_consensus_spectral_profiles(data['blocks'],cfg)
   for arm in ('F','E'):
    config=dataclasses.replace(cfg,subspace_consistency_lambda=0.0 if arm=='E' else cfg.subspace_consistency_lambda)
    profiles={n:{**v,'consistency_lambda':config.subspace_consistency_lambda} for n,v in profile.items()}
    ranks,stats=allocate_ranks(data['gradients'],config,candidate_energy=profiles)
    assert len(ranks)==72 and all(2<=v<=16 for v in ranks.values())
    used=sum(v*sum(data['gradients'][n]['grad'].shape) for n,v in ranks.items())
    assert used==755712
    if seed==42 and ranks!=old[task][arm]:
     raise RuntimeError(f'Seed-42 map mismatch: {task} {arm} changed={sum(ranks[n]!=old[task][arm][n] for n in ranks)}')
    out=ROOT/'rank_maps'/f'{task}_{arm}_head{seed}.json'
    if out.exists():
     assert json.loads(out.read_text())==ranks
    else:out.write_text(json.dumps(ranks,sort_keys=True,indent=2)+'\n')
    print(task,seed,arm,'changed_vs_seed42',sum(ranks[n]!=old[task][arm][n] for n in ranks),
          'map_sha256',hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':
 torch.set_num_threads(4);main()
