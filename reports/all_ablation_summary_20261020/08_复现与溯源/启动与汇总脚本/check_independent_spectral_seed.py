#!/usr/bin/env python3
"""Post-primary diagnostic: independent randomized-SVD seed for heldout B only."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/workspace/SpecCon-LoRA/experiments/independent_calibration_mechanism_20261019_v1')
OUT=Path('/workspace/SpecCon-LoRA/reports/independent_calibration_mechanism_20261019')
sys.path.insert(0,str(ROOT/'source_snapshot'))
sys.path.insert(0,'/workspace/SpecCon-LoRA/reproduction')
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, set_seed
from torch.utils.data import DataLoader
from speccon_lora.config import SpecConLoraConfig
from speccon_lora.calibration import collect_consensus_gradients
from speccon_lora.consensus import build_consensus_spectral_profiles
from analyze_independent_calibration_mechanism import controlled_residuals, corr, avg_sd, write_csv, module_type, sha


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    manifest=json.loads((ROOT/'manifest.json').read_text())
    primary=json.loads((OUT/'summary.json').read_text())
    dest=ROOT/'independent_spectral_seed_check'
    dest.mkdir(exist_ok=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    diagnostic=dict(stage='post-primary numerical sensitivity; not a new selected configuration',
        heldout_spectral_seed=103407, original_spectral_seed=3407,
        reason='Check whether shared randomized SVD probes inflate A/B recurrence correlations.',
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), source_sha256=sha(__file__),
        manifest_sha256=sha(ROOT/'manifest.json'), sample_ids_changed=False, dropout_seed_changed=False)
    (dest/'protocol.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(manifest['model_path'],local_files_only=True)
    records=[]
    summary={}
    for task in ['cola','mrpc','rte']:
        spec=manifest['tasks'][task]
        cfg=SpecConLoraConfig(**json.loads((ROOT/'runs'/task/'config.json').read_text()))
        cfg=dataclasses.replace(cfg,spectral_seed=103407,subspace_consistency_floor=0.0)
        raw=load_from_disk('/workspace/datasets/glue/'+task)['train'].select(spec['heldout_ids'])
        def tokenize(batch):
            d=tokenizer(*[batch[k] for k in spec['text_fields']],truncation=True,max_length=spec['max_length'])
            d['labels']=batch['label'];return d
        ds=raw.map(tokenize,batched=True,remove_columns=raw.column_names)
        set_seed(42)
        model=AutoModelForSequenceClassification.from_pretrained(manifest['model_path'],num_labels=2,
            problem_type='single_label_classification',local_files_only=True,dtype=torch.float32).cuda()
        h=hashlib.sha256()
        for n,v in model.state_dict().items():
            h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
        assert h.hexdigest()==json.loads((ROOT/'runs'/task/'DONE.json').read_text())['model_sha256']
        loader=DataLoader(ds,batch_size=32,shuffle=False,collate_fn=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8),
                          generator=torch.Generator().manual_seed(9001))
        set_seed(manifest['b_dropout_seed'])
        def loss_fn(current,batch):
            return current(**{k:v.cuda() for k,v in batch.items()}).loss
        _,grads,blocks,steps=collect_consensus_gradients(model,loader,loss_fn,cfg,average_distributed=False)
        assert steps==16
        bprofiles=build_consensus_spectral_profiles(blocks,cfg)
        old=torch.load(ROOT/'runs'/task/'heldout_raw.pt',map_location='cpu',weights_only=False)
        errors={n:float((v['grad']-old['gradients'][n]['grad']).abs().max())/max(float(old['gradients'][n]['grad'].abs().max()),1e-30)
                for n,v in grads.items()}
        assert max(errors.values())<1e-5, errors
        torch.save(dict(blocks=blocks,profiles=bprofiles,gradient_relative_errors=errors),dest/(task+'_b.pt'))
        del model,grads,old
        torch.cuda.empty_cache()
        for rep in range(5):
            a=torch.load(ROOT/'runs'/task/f'a{rep}_order0_raw.pt',map_location='cpu',weights_only=False)
            rows=[]
            for name,p in a['raw_profiles'].items():
                cross=[]
                for ba in a['blocks']:
                    for bb in blocks:
                        mat=(ba[name]['left'].T@bb[name]['left']).square()
                        cross.append(mat.cumsum(0).cumsum(1).diagonal()/torch.arange(1,17))
                recurrence=torch.stack(cross).mean(0).tolist()
                shape='x'.join(map(str,a['gradients'][name]['grad'].shape))
                for k in range(3,17):
                    rows.append(dict(k=k,shape=shape,module_type=module_type(name),consistency=float(p['consistency'][k-1]),
                        log_cumulative_energy=float(np.log(max(float(p['mean_energy'][:k].sum()),1e-30))),
                        recurrence=recurrence[k-1],b_consistency=float(bprofiles[name]['consistency'][k-1])))
            for outcome in ['recurrence','b_consistency']:
                for control in ['shape','module_type']:
                    x,y=controlled_residuals(rows,outcome,control)
                    records.append(dict(task=task,replicate=rep,outcome=outcome,control=control,
                                         partial_rank_correlation=corr(x,y)))
            del a
        t={'max_gradient_relative_difference':max(errors.values()),'predictivity':{}}
        for outcome in ['recurrence','b_consistency']:
            t['predictivity'][outcome]={}
            for control in ['shape','module_type']:
                stat=avg_sd([r['partial_rank_correlation'] for r in records if r['task']==task and r['outcome']==outcome and r['control']==control])
                stat['original_seed_mean']=primary['tasks'][task]['predictivity'][outcome][control]['mean']
                t['predictivity'][outcome][control]=stat
        summary[task]=t
        print(json.dumps({task:t}),flush=True)
    write_csv(OUT/'independent_spectral_seed.csv',records)
    (OUT/'independent_spectral_seed.json').write_text(json.dumps(summary,indent=2)+'\n')
    (dest/'DONE.json').write_text(json.dumps(dict(complete=True,additional_calibrations=3,
        artifacts={p.name:sha(p) for p in dest.glob('*') if p.is_file() and p.name!='DONE.json'}),indent=2)+'\n')


if __name__=='__main__':
    main()
