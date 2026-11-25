#!/usr/bin/env python3
"""Wait for preregistered runs and write observed C1/R1 tables."""
import csv
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,items):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=items[0].keys())
        writer.writeheader();writer.writerows(items)


def wait(plan,session):
    while True:
        paths=[Path(j['output'])/'metrics.json' for j in plan]
        if all(p.is_file() and json.loads(p.read_text()).get('state')=='complete' for p in paths):
            return
        if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError(f'{session} ended before all results completed')
        time.sleep(30)


def main():
    c1=json.loads((ROOT/'configs/c1_run_plan.json').read_text())
    r1=json.loads((ROOT/'configs/r1_run_plan.json').read_text())
    wait(c1,'speccon_c1_20261108')
    wait(r1,'speccon_r1_queue_20261108')
    crows=[]
    for job in c1:
        metrics=Path(job['output'])/'metrics.json';m=json.loads(metrics.read_text())
        spec=json.loads((ROOT/'configs'/f"{job['id']}.json").read_text())
        crows.append({'task':'mrpc','seed':spec['seed'],'arm':'F' if spec['lambda'] else 'E',
            'status':'observed','final_accuracy_percent':m['final_accuracy_percent'],
            'adapter_budget':m['adapter_parameters'],'final_step':m['optimizer_steps'],
            'metrics_path':str(metrics)})
    write(ROOT/'results/calibration_ref_bridge.csv',crows)
    pair=[]
    for seed in (42,777,2026):
        f=next(r for r in crows if r['seed']==seed and r['arm']=='F')
        e=next(r for r in crows if r['seed']==seed and r['arm']=='E')
        pair.append(float(f['final_accuracy_percent'])-float(e['final_accuracy_percent']))
    old=rows(PROJECT/'experiments/mrpc_calibration_robustness_v1/paired_results.csv')
    old_delta=[float(r['delta_consensus_minus_energy']) for r in old]
    rrows=[]
    maps=json.loads((ROOT/'data/rank_maps.json').read_text())
    for job in r1:
        seed=int(job['id'].rsplit('s',1)[1]);metrics=Path(job['output'])/'metrics.json'
        m=json.loads(metrics.read_text());f=json.loads(Path(maps['cola'][str(seed)]['F']['metrics']).read_text())
        rrows.append({'task':'cola','seed':seed,'permutation_seed':20261108,'status':'observed',
            'full_final_mcc_percent':f['final_score_percent'],
            'p1_final_mcc_percent':m['final_score_percent'],
            'delta_full_minus_p1':f['final_score_percent']-m['final_score_percent'],
            'adapter_budget':m['adapter_parameters'],'final_step':m['optimizer_steps'],
            'metrics_path':str(metrics)})
    write(ROOT/'results/global_rank_map_counterfactual.csv',rrows)
    paired=rows(ROOT/'results/paired_geometry_accuracy.csv')
    for row in paired:
        if row['task']=='cola' and row['contrast']=='F-P1':
            r=next(x for x in rrows if x['seed']==int(row['seed']))
            row['delta_dev_metric']=r['delta_full_minus_p1']
            row['status']='observed_historical_F_vs_new_P1'
            row['fixed_map_three_seed_protocol']='incomparable'
    write(ROOT/'results/paired_geometry_accuracy.csv',paired)
    report=[
        '# C1/R1 完成记录','',
        'MRPC A_ref 同缓存协议 Full−Energy 三种子差值：'+', '.join(f'{v:+.3f}' for v in pair)+'.',
        f'均值 {statistics.mean(pair):+.3f}；A0–A4 原 15 个有效内部配对均值 {statistics.mean(old_delta):+.3f}。',
        '这两个组均使用独立缓存流程；A_ref 与 A0–A4 的比较仍包含样本和顺序差异。','',
        'CoLA 原始 F−新 P1 逐种子差值：'+', '.join(f"{r['delta_full_minus_p1']:+.3f}" for r in rrows)+'.',
        f"均值 {statistics.mean(r['delta_full_minus_p1'] for r in rrows):+.3f}。",
        '几何差值见 geometry_map_scores.csv 和 paired_geometry_accuracy.csv；F 与 P1 为逐种子参考 map，不属于固定 A_ref map 的三种子性能重复。','',
        '所有数值来自固定末轮指标。原始指标路径在对应 CSV 中；没有按验证集结果挑选置换或种子。'
    ]
    (ROOT/'analysis/C1_R1_COMPLETION.md').write_text('\n'.join(report)+'\n')
    print('finalized C1 and R1',flush=True)


if __name__=='__main__':main()
