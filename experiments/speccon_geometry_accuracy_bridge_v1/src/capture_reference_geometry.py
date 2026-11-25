#!/usr/bin/env python3
"""Collect frozen A/B gradients for the preregistered geometry bridge."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
FROZEN = PROJECT / 'experiments/independent_calibration_mechanism_20261019_v1/source_snapshot'
sys.path.insert(0, str(FROZEN))

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, set_seed
from speccon_lora.config import SpecConLoraConfig
from speccon_lora.calibration import collect_consensus_gradients


def digest_state(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_task(task, partitions_only=False):
    ids = json.loads((ROOT / 'data/sample_ids_A_B1_B2.json').read_text())[task]
    old = json.loads((PROJECT / 'experiments/independent_calibration_mechanism_20261019_v1/manifest.json').read_text())
    spec = old['tasks'][task]
    assert ids['dataset_fingerprint'] == spec['dataset_fingerprint']
    model_path = old['model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    raw = load_from_disk('/workspace/datasets/glue/' + task)['train']
    def tokenize(batch):
        values = [batch[k] for k in spec['text_fields']]
        output = tokenizer(*values, truncation=True, max_length=spec['max_length'])
        output['labels'] = batch['label']
        return output
    ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, problem_type='single_label_classification',
        local_files_only=True, dtype=torch.float32).cuda()
    state_hash = digest_state(model)
    config = SpecConLoraConfig(
        target_modules=['query','key','value','dense'], target_prefix='roberta.encoder.layer',
        base_rank=8, r_min=2, r_max=16, lora_alpha=2, calibration_steps=16,
        calibration_blocks=4, use_subspace_consistency=True,
        subspace_consistency_lambda=spec['consistency_lambda'],
        subspace_consistency_floor=0.05, spectral_oversample=8,
        spectral_power_iterations=1, spectral_seed=3407,
        init_method='svd_a_zero_b', init_scale=1.0, scaling_mode='constant',
        rank_budget_mode='param', param_budget=755712, strict_param_budget=True,
        rank_score_gamma=1.2, attention_rank_prior=1.0)
    work = ROOT / 'caches' / task
    work.mkdir(parents=True, exist_ok=True)
    arms = tuple(f'A_partition_{seed}' for seed in (20261109,20261110,20261111)) if partitions_only else ('A','B1','B2')
    for arm in arms:
        output = work / f'{arm}_raw.pt'
        if output.exists():
            print(f'{task}/{arm}: existing cache', flush=True)
            continue
        if arm.startswith('A_partition_'):
            import numpy as np
            selected = np.random.default_rng(int(arm.rsplit('_',1)[1])).permutation(ids['A']).tolist()
        else:
            selected = ids[arm]
        cfg = dataclasses.replace(config, calibration_steps=len(selected)//32)
        set_seed(7001 if arm.startswith('A') else 8001)
        loader = DataLoader(ds.select(selected), batch_size=32, shuffle=False,
            collate_fn=collator, generator=torch.Generator().manual_seed(9001))
        def loss_fn(current, batch):
            return current(**{k:v.cuda() for k,v in batch.items()}).loss
        start = time.time()
        names, gradients, blocks, steps = collect_consensus_gradients(
            model, loader, loss_fn, cfg, average_distributed=False)
        assert len(names)==72 and len(gradients)==72 and len(blocks)==4
        assert steps==len(selected)//32
        payload={'task':task,'arm':arm,'ids':selected,'dropout_seed':7001 if arm.startswith('A') else 8001,
            'model_state_sha256':state_hash,'gradients':gradients,'blocks':blocks,
            'steps':steps,'config':dataclasses.asdict(cfg),'seconds':time.time()-start}
        temp = output.with_suffix('.tmp')
        torch.save(payload,temp)
        temp.replace(output)
        with (ROOT/'run_manifest.jsonl').open('a') as f:
            f.write(json.dumps({'stage':'G1_capture','task':task,'arm':arm,'state':'completed',
                'path':str(output),'model_state_sha256':state_hash,'steps':steps,
                'seconds':payload['seconds'],'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\n')
        print(f'{task}/{arm}: captured {payload["seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',nargs='+',choices=['mrpc','cola','rte'],default=['mrpc','cola','rte'])
    parser.add_argument('--partitions-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    for task in args.tasks: run_task(task,args.partitions_only)
