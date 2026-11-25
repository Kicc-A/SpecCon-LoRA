#!/usr/bin/env python3
"""Capture only A_ref gradients with a preregistered model/head seed."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/SpecCon-LoRA/experiments/speccon_geometry_accuracy_bridge_v1')
sys.path.insert(0,str(ROOT/'source_snapshot'))
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,set_seed
from speccon_lora.config import SpecConLoraConfig
from speccon_lora.calibration import collect_consensus_gradients
from extended_ablation_support import tensor_hash

FIELDS={'cola':['sentence'],'mrpc':['sentence1','sentence2'],'rte':['sentence1','sentence2']}
LENGTH={'cola':320,'mrpc':128,'rte':256}
LAMBDA={'cola':1.5,'mrpc':1.5,'rte':0.5}

def state_hash(model):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():
  h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=list(FIELDS),required=True)
 p.add_argument('--model-seed',type=int,choices=[777,2026],required=True);a=p.parse_args()
 out=ROOT/'caches'/f'{a.task}_head{a.model_seed}_A_raw.pt'
 if out.exists(): raise RuntimeError(f'Refusing overwrite: {out}')
 ids=json.loads((BASE/'data/sample_ids_A_B1_B2.json').read_text())[a.task]['A']
 assert ids==list(range(512))
 tokenizer=AutoTokenizer.from_pretrained('/workspace/Models/RoBERTa-base',local_files_only=True)
 raw=load_from_disk('/workspace/datasets/glue/'+a.task)['train']
 def tokenize(batch):
  x=tokenizer(*[batch[k] for k in FIELDS[a.task]],truncation=True,max_length=LENGTH[a.task]);x['labels']=batch['label'];return x
 ds=raw.map(tokenize,batched=True,remove_columns=raw.column_names)
 collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8)
 set_seed(a.model_seed)
 model=AutoModelForSequenceClassification.from_pretrained('/workspace/Models/RoBERTa-base',num_labels=2,
     problem_type='single_label_classification',local_files_only=True,dtype=torch.float32).cuda()
 head=tensor_hash([(n,p) for n,p in model.named_parameters() if 'classifier' in n])
 cfg=SpecConLoraConfig(target_modules=['query','key','value','dense'],target_prefix='roberta.encoder.layer',
     base_rank=8,r_min=2,r_max=16,lora_alpha=2,calibration_steps=16,calibration_blocks=4,
     use_subspace_consistency=True,subspace_consistency_lambda=LAMBDA[a.task],
     subspace_consistency_floor=0.05,spectral_oversample=8,spectral_power_iterations=1,
     spectral_seed=3407,init_method='svd_a_zero_b',init_scale=1.0,scaling_mode='constant',
     rank_budget_mode='param',param_budget=755712,strict_param_budget=True,
     rank_score_gamma=1.2,attention_rank_prior=1.0)
 set_seed(7001)
 loader=DataLoader(ds.select(ids),batch_size=32,shuffle=False,collate_fn=collator,
                   generator=torch.Generator().manual_seed(9001))
 def loss_fn(current,batch):return current(**{k:v.cuda() for k,v in batch.items()}).loss
 start=time.time()
 names,grads,blocks,steps=collect_consensus_gradients(model,loader,loss_fn,cfg,average_distributed=False)
 assert len(names)==len(grads)==72 and len(blocks)==4 and steps==16
 payload=dict(task=a.task,arm='A',ids=ids,dropout_seed=7001,model_init_seed=a.model_seed,
              model_state_sha256=state_hash(model),classifier_sha256=head,
              gradients=grads,blocks=blocks,steps=steps,config=dataclasses.asdict(cfg),seconds=time.time()-start)
 tmp=out.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(out)
 print(json.dumps(dict(task=a.task,seed=a.model_seed,head_sha256=head,state_sha256=payload['model_state_sha256'],
                       output=str(out),seconds=payload['seconds'],gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))),flush=True)
if __name__=='__main__':
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 main()
