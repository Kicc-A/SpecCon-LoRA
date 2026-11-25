#!/usr/bin/env python3
"""Audit all matched-head F/E pairs and publish every outcome."""
import csv
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/workspace/SpecCon-LoRA/experiments/speccon_geometry_accuracy_bridge_v1')
OLD=json.loads((BASE/'data/reference_A_fixed_rank_maps.json').read_text())

def write(path,rows,fields):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def run(task,arm,seed):
 if seed==42:
  d=BASE/'runs'/('c1_mrpc_ref_lambda'+('1.5' if arm=='F' else '0.0')+'_s42') if task=='mrpc' else BASE/'runs'/f'fixed_{task}_{arm}_s42'
 else:d=ROOT/'runs'/f'{task}_{arm}_s{seed}'
 metric=d/'metrics.json'
 if not metric.exists():return None
 m=json.loads(metric.read_text())
 if m.get('state')!='complete':return None
 ranks=json.loads((d/'allocation.json').read_text())['rank_pattern']
 expected=OLD[task][arm] if seed==42 else json.loads((ROOT/'rank_maps'/f'{task}_{arm}_head{seed}.json').read_text())
 assert ranks==expected and m['adapter_parameters']==755712
 assert m['optimizer_steps']=={'cola':11256,'mrpc':6555,'rte':4914}[task]
 state=json.loads((d/'initial_state.json').read_text())
 rng=json.loads((d/'training_rng.json').read_text())
 head=state['classifier_sha256']
 score=m['final_score_percent'] if task=='cola' else m['final_accuracy_percent']
 return dict(task=task,arm=arm,seed=seed,score=score,head=head,cpu_rng=rng['cpu_sha256'],
             cuda_rng=rng['cuda_sha256'],changed_rank_modules=sum(ranks[n]!=OLD[task][arm][n] for n in ranks),
             metrics_path=str(metric))

rows=[];paired=[];summary=[]
for task in ('cola','mrpc','rte'):
 ds=[]
 for seed in (42,777,2026):
  f=run(task,'F',seed);e=run(task,'E',seed)
  for x in (f,e):
   if x:rows.append(x)
  if f and e:
   assert f['head']==e['head'] and f['cpu_rng']==e['cpu_rng'] and f['cuda_rng']==e['cuda_rng']
   ds.append(f['score']-e['score'])
  paired.append(dict(task=task,seed=seed,full=f['score'] if f else '',energy=e['score'] if e else '',
     delta=f['score']-e['score'] if f and e else '',status='observed' if f and e else 'missing',
     full_changed_modules=f['changed_rank_modules'] if f else '',
     energy_changed_modules=e['changed_rank_modules'] if e else ''))
 summary.append(dict(task=task,n=len(ds),delta_mean=statistics.mean(ds) if len(ds)==3 else '',
    delta_sd=statistics.stdev(ds) if len(ds)==3 else '',wins=sum(v>0 for v in ds),
    ties=sum(v==0 for v in ds),losses=sum(v<0 for v in ds),
    status='observed' if len(ds)==3 else 'missing'))
write(ROOT/'results/per_seed.csv',rows,['task','arm','seed','score','head','cpu_rng','cuda_rng','changed_rank_modules','metrics_path'])
write(ROOT/'results/paired.csv',paired,['task','seed','full','energy','delta','status','full_changed_modules','energy_changed_modules'])
write(ROOT/'results/summary.csv',summary,['task','n','delta_mean','delta_sd','wins','ties','losses','status'])
lines=['# 分类头匹配的校准消融','',f'完整训练单元：{len(rows)}/18；其中 seed 42 的 6 个单元复用固定 map 桥接。','',
 '| 任务 | F−E 均值 ± 配对 SD | 胜/平/负 | 状态 |','| --- | ---: | ---: | --- |']
for r in summary:
 v=f"{r['delta_mean']:+.3f} ± {r['delta_sd']:.3f}" if r['status']=='observed' else '待完成'
 lines.append(f"| {r['task']} | {v} | {r['wins']}/{r['ties']}/{r['losses']} | {r['status']} |")
lines.extend(['','这里的 seed 777/2026 校准梯度、分类头初态和训练 seed 相匹配；seed 42 则复用之前的相同协议结果。','所有条件使用同一 A_ref 512 样本、755,712 适配器参数、任务内相同训练配方和同 seed 训练 RNG；取固定末轮。','与旧的即时校准主表及固定 seed42 头的跨 seed map 桥接分开报告，不把差异归因于单一因素。',''])
(ROOT/'analysis/STATUS.md').write_text('\n'.join(lines))
print('complete',len(rows),'/18')
