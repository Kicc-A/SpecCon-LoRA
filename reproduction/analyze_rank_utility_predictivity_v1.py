#!/usr/bin/env python3
"""Analyze all pre-registered MRPC swaps against one matched reference."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/workspace/SpecCon-LoRA/experiments/rank_utility_predictivity_v1')
METHODS = ('raw_spectral_energy','energy_only_utility','consensus_adjusted_utility')
LABELS = ('Raw energy','Energy utility','Consensus utility')


def read(path):
    return json.loads(Path(path).read_text())


def csv_out(path, rows, fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def main():
    prereg=read(ROOT/'pre_registered_swaps.json')
    all_jobs=read(ROOT/'manifest.json')['jobs']
    rows=[]
    for job in all_jobs:
        folder=Path(job['output'])
        if not (folder/'metrics.json').exists():
            continue
        metric=read(folder/'metrics.json')
        assert metric['state']=='complete' and metric['final']==metric['selected']
        assert metric['epochs']==57 and metric['adapter_parameters']==755712
        allocation=read(folder/'allocation.json')
        assert allocation['rank_report']['budget_used']==755712
        rows.append({'run_id':job['id'],'swap_id':job['spec']['name'],
            'final_accuracy_percent':100*metric['final']['splits']['validation']['accuracy'],
            'epochs':metric['epochs'],'adapter_budget':metric['adapter_parameters'],
            'output':str(folder)})
    csv_out(ROOT/'swap_runs.csv',rows,
        ['run_id','swap_id','final_accuracy_percent','epochs','adapter_budget','output'])
    byid={row['swap_id']:row for row in rows}
    baseline=byid.get('reference')
    observed=[]
    for item in prereg['selected']:
        run=byid.get(item['swap_id'])
        if run is None or baseline is None:
            continue
        delta=run['final_accuracy_percent']-baseline['final_accuracy_percent']
        observed.append({'swap_id':item['swap_id'],
            'selection_category':item['selection_category'],
            'receiver_module':item['receiver_module'],'donor_module':item['donor_module'],
            'observed_accuracy_delta':delta,
            'reference_accuracy':baseline['final_accuracy_percent'],
            'swap_accuracy':run['final_accuracy_percent'],
            **{method+'_predicted_delta':item['predicted_delta'][method] for method in METHODS}})
    csv_out(ROOT/'predicted_vs_observed.csv',observed,
        ['swap_id','selection_category','receiver_module','donor_module',
         'observed_accuracy_delta','reference_accuracy','swap_accuracy']+
        [method+'_predicted_delta' for method in METHODS])
    complete=len(observed)==12
    stats=[]
    if complete:
        y=np.array([row['observed_accuracy_delta'] for row in observed])
        for method in METHODS:
            x=np.array([row[method+'_predicted_delta'] for row in observed])
            rho,p=spearmanr(x,y)
            top=np.argsort(x)[-3:]
            stats.append({'method':method,'spearman_rho':float(rho),'spearman_p_exploratory':float(p),
                'direction_accuracy':float(np.mean(np.sign(x)==np.sign(y))),
                'top3_actual_mean_delta':float(y[top].mean()),
                'top3_ids':[observed[i]['swap_id'] for i in top]})
    fig,ax=plt.subplots(figsize=(7,4))
    if complete:
        ax.bar(range(3),[s['spearman_rho'] for s in stats],color=['#7c8790','#4784a7','#c37435'])
        ax.axhline(0,color='.5',lw=.8)
        ax.set_xticks(range(3),LABELS);ax.set_ylim(-1,1)
    else:
        ax.text(.5,.5,f'Waiting for matched runs: {len(observed)}/12',
            transform=ax.transAxes,ha='center',va='center')
        ax.set_xticks([]);ax.set_yticks([])
    ax.set_ylabel('Spearman correlation with observed Accuracy change')
    fig.tight_layout();fig.savefig(ROOT/'utility_spearman_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for ax,method,label in zip(axes,METHODS,LABELS):
        if observed:
            x=np.array([r[method+'_predicted_delta'] for r in observed])
            y=np.array([r['observed_accuracy_delta'] for r in observed])
            ax.scatter(x,y,s=26)
            for xx,yy,row in zip(x,y,observed):
                ax.annotate(row['swap_id'].replace('swap',''),(xx,yy),fontsize=7)
        ax.axhline(0,color='.6',lw=.8);ax.axvline(0,color='.6',lw=.8)
        ax.set_title(label);ax.set_xlabel('Predicted utility change')
    axes[0].set_ylabel('Observed final Accuracy change (pp)')
    fig.tight_layout();fig.savefig(ROOT/'predicted_vs_observed_scatter.png',dpi=180);plt.close(fig)
    lines=['# EXP-02 Rank Utility Predictivity', '',
        f'固定候选：{len(prereg["selected"])} / {prereg["all_feasible_count"]} 个可行交换；'
        f'匹配参考已完成：{"是" if baseline else "否"}；交换已完成 {len(observed)}/12。',
        '预注册文件在任何交换训练前生成；全部12组都进入分析，不按结果筛选。'
        '原 near-uniform 参考的校准 ID 与 A0 不同，已排除复用。',
        '每个交换只改变同形状两模块的 rank（+1 / −1），参数预算严格为755,712。', '',
        '## 观察事实', '']
    if complete:
        lines += ['|评分|Spearman ρ|预测方向命中率|Top-3 实际平均收益 (pp)|',
                  '|---|---:|---:|---:|']
        for s,label in zip(stats,LABELS):
            lines.append(f'|{label}|{s["spearman_rho"]:+.3f}|{s["direction_accuracy"]:.1%}|'
                f'{s["top3_actual_mean_delta"]:+.3f}|')
        gap=stats[2]['spearman_rho']-stats[1]['spearman_rho']
        lines += ['',f'Consensus 相对 Energy-only 的 Spearman 差值：{gap:+.3f}。',
            '高预测收益但真实 Accuracy 下降的候选：']
        misses=[r['swap_id'] for r in observed if r['consensus_adjusted_utility_predicted_delta']>0
                and r['observed_accuracy_delta']<0]
        lines.append('、'.join(misses) if misses else '无。')
    else:
        lines.append('训练仍在排队或进行，尚不能计算完整12组的排序相关和 Top-k。')
    lines += ['', '## 可能机制解释', '',
        '预测效用仅来自预训练前冻结的 A0 梯度谱；真实收益来自完整下游训练。'
        '若两者排序不一致，模块间共同优化、校准误差或单种子 Accuracy 离散性都可能参与。', '',
        '## 目前不能确定', '',
        '12个交换共享同一参考训练，观测收益相关，p值只作探索性描述。'
        '只有在完整第一阶段显示预先定义的可解释信号时，才考虑固定候选的多种子扩展。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'analysis.json').write_text(json.dumps({'complete':complete,'reference_complete':baseline is not None,
        'swaps_complete':len(observed),'statistics':stats},indent=2)+'\n')


if __name__=='__main__':
    main()
