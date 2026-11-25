#!/usr/bin/env python3
"""Write the preregistered comparison when all three formal runs finish."""
import csv
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BRIDGE=ROOT.parent/'speccon_geometry_accuracy_bridge_v1'


def main():
    paths=[ROOT/f'runs/ensemble_s{s}/metrics.json' for s in (42,777,2026)]
    while not all(p.exists() and json.loads(p.read_text()).get('state')=='complete' for p in paths):
        if subprocess.run(['tmux','has-session','-t','mrpc_order_ensemble_20261109'],
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError('Training controller ended without all three completed runs')
        time.sleep(30)
    rows=[]
    for seed,p in zip((42,777,2026),paths):
        ens=json.loads(p.read_text())
        f=json.loads((BRIDGE/f'runs/c1_mrpc_ref_lambda1.5_s{seed}/metrics.json').read_text())
        e=json.loads((BRIDGE/f'runs/c1_mrpc_ref_lambda0.0_s{seed}/metrics.json').read_text())
        assert ens['adapter_parameters']==f['adapter_parameters']==e['adapter_parameters']==755712
        initial=json.loads((ROOT/f'runs/ensemble_s{seed}/initial_state.json').read_text())
        base_initial=json.loads((BRIDGE/f'runs/c1_mrpc_ref_lambda1.5_s{seed}/initial_state.json').read_text())
        rng=json.loads((ROOT/f'runs/ensemble_s{seed}/training_rng.json').read_text())
        base_rng=json.loads((BRIDGE/f'runs/c1_mrpc_ref_lambda1.5_s{seed}/training_rng.json').read_text())
        assert initial['classifier_sha256']==base_initial['classifier_sha256']
        assert rng['cpu_sha256']==base_rng['cpu_sha256'] and rng['cuda_sha256']==base_rng['cuda_sha256']
        assert ens['optimizer_steps']==f['optimizer_steps']==6555
        rows.append({'seed':seed,'ensemble_accuracy':ens['final_accuracy_percent'],
            'full_accuracy':f['final_accuracy_percent'],'energy_accuracy':e['final_accuracy_percent'],
            'ensemble_minus_full':ens['final_accuracy_percent']-f['final_accuracy_percent'],
            'ensemble_minus_energy':ens['final_accuracy_percent']-e['final_accuracy_percent'],
            'adapter_budget':755712,'final_step':6555,'head_and_training_rng_matched':True,
            'metrics_path':str(p)})
    with (ROOT/'results/paired_final_accuracy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    dif=[r['ensemble_minus_full'] for r in rows]
    geometry=list(csv.DictReader((ROOT/'results/geometry_scores.csv').open()))
    baseline=list(csv.DictReader((BRIDGE/'results/geometry_fixed_map_scores.csv').open()))
    q=[]
    for r in geometry:
        f=next(x for x in baseline if x['task']=='mrpc' and x['map']=='F' and x['probe']==r['probe'])
        q.append((r['probe'],float(r['q_geom'])-float(f['q_geom'])))
    report=['# MRPC 等权顺序共识实验结果','',
        '预注册：原顺序及三种固定重排的共识加权候选能量等权平均；只测试这一张 rank map，不按 B 几何或验证集挑选。',
        '相对 A_ref Full 改变 19/72 个模块 rank，严格维持 755,712 个适配器参数。','',
        '| seed | Ensemble Accuracy | Full Accuracy | Ensemble−Full | Ensemble−Energy |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        report.append(f"| {r['seed']} | {r['ensemble_accuracy']:.3f} | {r['full_accuracy']:.3f} | {r['ensemble_minus_full']:+.3f} | {r['ensemble_minus_energy']:+.3f} |")
    report += ['',f"主对照均值 {statistics.mean(dif):+.3f} 点；胜/平/负 {sum(x>0 for x in dif)}/{sum(x==0 for x in dif)}/{sum(x<0 for x in dif)}。",
        '独立 B 几何指标相对 Full：'+', '.join(f'{b} ΔQ_geom {v:+.5f}' for b,v in q)+'。',
        '所有指标取固定末轮；同 seed 分类头初态、训练 CPU/CUDA RNG、预算与最终步数均已核对。',
        '新方法同时使用原 A_ref 及三次重排的校准信息，因此额外校准计算成本不可忽略；顺序变化也改变样本与 dropout 掩码的对应。',
        '若增益不一致，应如实报告，不将几何指标当作最终精度的充分预测。','']
    (ROOT/'results/REPORT.md').write_text('\n'.join(report))
    print('finalized',flush=True)


if __name__=='__main__':main()
