#!/usr/bin/env python3
"""Audit and report the preregistered fixed A_ref F/E/S1/P1 contrasts."""
import csv
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'configs/fixed_map_completion_plan.json').read_text())
MAPS=json.loads((ROOT/'data/reference_A_fixed_rank_maps.json').read_text())
GEOM={}
with (ROOT/'results/geometry_fixed_map_scores.csv').open() as f:
    for row in csv.DictReader(f): GEOM[(row['task'],row['map'],row['probe'])]=row

def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def load_run(task,arm,seed):
    if task=='mrpc' and arm in ('F','E'):
        name=f'c1_mrpc_ref_lambda{1.5 if arm=="F" else 0.0}_s{seed}'
    else:name=f'fixed_{task}_{arm}_s{seed}'
    path=ROOT/'runs'/name
    metrics=path/'metrics.json'
    if not metrics.exists(): return None
    value=json.loads(metrics.read_text())
    if value.get('state')!='complete':return None
    allocation=json.loads((path/'allocation.json').read_text())
    assert allocation['rank_pattern']==MAPS[task][arm],(task,arm,seed)
    assert value['adapter_parameters']==755712
    expected_step={'cola':11256,'mrpc':6555,'rte':63*int((2490+31)//32)}[task]
    assert value['optimizer_steps']==expected_step,(task,arm,seed,value['optimizer_steps'],expected_step)
    state=json.loads((path/'initial_state.json').read_text())
    rng=json.loads((path/'training_rng.json').read_text())
    head=state.get('classifier_sha256') or state.get('base_classifier_sha256')
    assert head
    score=value['final_score_percent'] if task=='cola' else value['final_accuracy_percent']
    return dict(task=task,map=arm,seed=seed,score=score,head=head,cpu_rng=rng['cpu_sha256'],
                cuda_rng=rng['cuda_sha256'],optimizer_steps=value['optimizer_steps'],
                metrics_path=str(metrics),run_dir=str(path))

rows=[]; cache={}; issues=[]
for task in ('cola','mrpc','rte'):
    for seed in (42,777,2026):
        for arm in ('F','E','S1','P1'):
            data=load_run(task,arm,seed);cache[(task,arm,seed)]=data
            if data:rows.append(data)
            else:issues.append(f'missing {task} {arm} {seed}')
        observed=[cache[(task,arm,seed)] for arm in ('F','E','S1','P1') if cache[(task,arm,seed)]]
        if len(observed)>1:
            assert len({r['head'] for r in observed})==1,(task,seed,'head mismatch')
            assert len({r['cpu_rng'] for r in observed})==1,(task,seed,'CPU RNG mismatch')
            assert len({r['cuda_rng'] for r in observed})==1,(task,seed,'CUDA RNG mismatch')
write_csv(ROOT/'results/fixed_map_per_seed_validation.csv',rows,
          ['task','map','seed','score','head','cpu_rng','cuda_rng','optimizer_steps','metrics_path','run_dir'])
paired=[]
for task in ('cola','mrpc','rte'):
    for arm in ('S1','E','P1'):
        for seed in (42,777,2026):
            f=cache[(task,'F',seed)];x=cache[(task,arm,seed)]
            for probe in ('B1','B2'):
                gf=GEOM[(task,'F',probe)];gx=GEOM[(task,arm,probe)]
                paired.append(dict(task=task,contrast=f'F-{arm}',seed=seed,probe=probe,
                    delta_q_geom=float(gf['q_geom'])-float(gx['q_geom']),
                    delta_q_grad=float(gf['q_grad'])-float(gx['q_grad']),
                    full_score=f['score'] if f else '',control_score=x['score'] if x else '',
                    delta_dev=f['score']-x['score'] if f and x else '',
                    metric='MCC x100' if task=='cola' else 'Accuracy x100',
                    status='observed' if f and x else 'missing'))
write_csv(ROOT/'results/fixed_map_paired_geometry_accuracy.csv',paired,
          ['task','contrast','seed','probe','delta_q_geom','delta_q_grad','full_score','control_score','delta_dev','metric','status'])
summary=[]
for task in ('cola','mrpc','rte'):
    for arm in ('S1','E','P1'):
        d=[cache[(task,'F',s)]['score']-cache[(task,arm,s)]['score']
           for s in (42,777,2026) if cache[(task,'F',s)] and cache[(task,arm,s)]]
        q=[float(GEOM[(task,'F',b)]['q_geom'])-float(GEOM[(task,arm,b)]['q_geom']) for b in ('B1','B2')]
        summary.append(dict(task=task,contrast=f'F-{arm}',delta_q_geom_B1=q[0],delta_q_geom_B2=q[1],
            delta_dev_mean=statistics.mean(d) if len(d)==3 else '',
            delta_dev_paired_sd=statistics.stdev(d) if len(d)==3 else '',
            wins=sum(x>0 for x in d),ties=sum(x==0 for x in d),losses=sum(x<0 for x in d),
            n=len(d),status='observed' if len(d)==3 else 'missing'))
write_csv(ROOT/'results/fixed_map_contrast_summary.csv',summary,
          ['task','contrast','delta_q_geom_B1','delta_q_geom_B2','delta_dev_mean','delta_dev_paired_sd','wins','ties','losses','n','status'])
lines=['# 固定 A_ref rank map 补全状态','',f'已完成 {len(rows)}/36 个 task × map × seed 单元（其中 6 个 MRPC F/E 为先前 C1 结果）。','',
       '| 任务 | 对照 | ΔQ_geom B1/B2 | ΔDev 均值 ± 配对 SD | 胜/平/负 | 状态 |',
       '| --- | --- | --- | --- | --- | --- |']
for r in summary:
    delta=f"{r['delta_dev_mean']:+.3f} ± {r['delta_dev_paired_sd']:.3f}" if r['status']=='observed' else '待完成'
    lines.append(f"| {r['task']} | {r['contrast']} | {r['delta_q_geom_B1']:+.5f} / {r['delta_q_geom_B2']:+.5f} | {delta} | {r['wins']}/{r['ties']}/{r['losses']} | {r['status']} |")
lines.extend(['','所有性能取固定末轮；每任务按相同训练 seed 配对。几何指标来自同一固定 A_ref map 的 B1/B2。','',
              f'缺失单元：{len(issues)}。'])
if issues: lines.extend(['']+['- '+i for i in issues])
(ROOT/'analysis/FIXED_MAP_COMPLETION.md').write_text('\n'.join(lines)+'\n')
if len(rows)==36:
    tex=['% Fixed A_ref rank maps: actual completed results only',
         r'\begin{tabular}{llrrrr}',
         r'\hline',
         r'Task & Contrast & $\Delta Q_{\rm geom}(B1)$ & $\Delta Q_{\rm geom}(B2)$ & $\Delta$Dev & W/T/L \\',
         r'\hline']
    for r in summary:
        tex.append(f"{r['task']} & {r['contrast']} & {r['delta_q_geom_B1']:+.5f} & "
                   f"{r['delta_q_geom_B2']:+.5f} & "
                   f"{r['delta_dev_mean']:+.3f} $\\pm$ {r['delta_dev_paired_sd']:.3f} & "
                   f"{r['wins']}/{r['ties']}/{r['losses']} " + r'\\')
    tex.extend([r'\hline',r'\end{tabular}'])
    (ROOT/'analysis/FIXED_MAP_TABLE_FOR_PAPER.tex').write_text('\n'.join(tex)+'\n')
print('completed',len(rows),'of 36; missing',len(issues))
if issues: print('\n'.join(issues))
