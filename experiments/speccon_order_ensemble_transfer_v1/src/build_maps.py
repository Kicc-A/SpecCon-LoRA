#!/usr/bin/env python3
"""Build the fixed four-order ensemble map, validating against MRPC pilot."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/SpecCon-LoRA/experiments/speccon_geometry_accuracy_bridge_v1')
sys.path.insert(0,str(BASE/'fixed_map_source'))
from speccon_lora.config import SpecConLoraConfig
from speccon_lora.consensus import build_consensus_spectral_profiles
from speccon_lora.rank_allocator import allocate_ranks
PILOT=Path('/workspace/SpecCon-LoRA/experiments/mrpc_consensus_order_ensemble_v1/configs/ensemble_rank_map.json')
for task in ('mrpc','cola','rte'):
 caches=[]
 for arm in ('A','A_partition_20261109','A_partition_20261110','A_partition_20261111'):
  path=BASE/'caches'/task/f'{arm}_raw.pt'
  if not path.exists():raise FileNotFoundError(path)
  caches.append(torch.load(path,map_location='cpu',weights_only=False))
 cfg=SpecConLoraConfig(**caches[0]['config'])
 assert all(set(c['ids'])==set(range(512)) for c in caches)
 assert all(c['model_state_sha256']==caches[0]['model_state_sha256'] for c in caches)
 profiles=[build_consensus_spectral_profiles(c['blocks'],cfg) for c in caches]
 combined={}
 for n in caches[0]['gradients']:
  combined[n]=sum(p[n]['mean_energy']*p[n]['consistency'].pow(cfg.subspace_consistency_lambda) for p in profiles)/4
 ranks,stats=allocate_ranks(caches[0]['gradients'],cfg,candidate_energy=combined)
 assert len(ranks)==72 and all(2<=r<=16 for r in ranks.values())
 assert sum(r*sum(caches[0]['gradients'][n]['grad'].shape) for n,r in ranks.items())==755712
 if task=='mrpc':
  pilot=json.loads(PILOT.read_text())['ranks']
  if ranks!=pilot:raise RuntimeError(f'MRPC pilot map mismatch: {sum(ranks[n]!=pilot[n] for n in ranks)} modules')
 out=ROOT/'configs'/f'{task}_ensemble_rank_map.json'
 out.write_text(json.dumps(ranks,indent=2,sort_keys=True)+'\n')
 old=json.loads((BASE/'data/reference_A_fixed_rank_maps.json').read_text())[task]['F']
 print(task,'changed_vs_F',sum(ranks[n]!=old[n] for n in ranks),'sha256',hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
