#!/usr/bin/env python3
"""Descriptive, replicate-level analysis of the frozen calibration campaign."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TASKS = ['cola', 'mrpc', 'rte']
COLORS = {'energy':'#6d7482', 'consistency':'#247ba0', 'near_uniform':'#d59a36'}
LABELS = {'energy':r'Block energy ($\lambda=0$)', 'consistency':'Consistency', 'near_uniform':'Near-uniform'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def module_type(name):
    return re.sub(r'^roberta\.encoder\.layer\.\d+\.', '', name)


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0,1])


def standardized_rank(v):
    x = rankdata(v)
    return (x-x.mean()) / max(float(x.std()), 1e-12)


def controlled_residuals(rows, outcome, group_by='shape', fixed_k=True):
    groups = {}
    for r in rows:
        key = ((r['k'],) if fixed_k else ()) + (r[group_by],)
        groups.setdefault(key, []).append(r)
    xs, ys = [], []
    for group in groups.values():
        if len(group) < 4:
            continue
        x = standardized_rank([r['consistency'] for r in group])
        y = standardized_rank([r[outcome] for r in group])
        energy = standardized_rank([r['log_cumulative_energy'] for r in group])
        z = np.column_stack([np.ones(len(group)), energy])
        xs.extend(x-z@np.linalg.lstsq(z,x,rcond=None)[0])
        ys.extend(y-z@np.linalg.lstsq(z,y,rcond=None)[0])
    return np.asarray(xs), np.asarray(ys)


def avg_sd(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return {'mean':float(a.mean()) if len(a) else None,
            'sd':float(a.std(ddof=1)) if len(a)>1 else None,
            'values':a.tolist()}


def distance(a, b, modules):
    return sum(modules[n]['cost']*abs(a[n]-b[n]) for n in a)/(2*755712)


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text())
    runs = {}
    audit = {'complete_tasks': [], 'hash_mismatches': [], 'budget_checks':0,
             'ab_id_disjoint':True, 'model_unchanged':True, 'checks':{}}
    for task in TASKS:
        done = json.loads((root/'runs'/task/'DONE.json').read_text())
        assert done['complete'] and done['model_unchanged']
        audit['complete_tasks'].append(task)
        for name, expected in done['artifacts'].items():
            if sha(root/'runs'/task/name) != expected:
                audit['hash_mismatches'].append(f'{task}/{name}')
        rows = [json.loads(p.read_text()) for p in sorted((root/'runs'/task).glob('a*_metrics.json'))]
        assert len(rows) == 10
        runs[task] = rows
        heldout = set(manifest['tasks'][task]['heldout_ids'])
        assert all(not heldout & set(order['ids']) for rep in manifest['tasks'][task]['replicates'] for order in rep['orders'])
        assert len({r['model_sha256'] for r in rows}) == 1
        for r in rows:
            assert r['max_orthogonality_error'] < 1e-4
            assert r['max_spectral_energy_relative_error'] < 1e-4
            for a in r['allocations'].values():
                assert a['budget_used'] == 755712
                assert 0 <= a['left_capture'] <= 1 and 0 <= a['right_capture'] <= 1
                audit['budget_checks'] += 1
        audit['checks'][task] = dict(max_orthogonality_error=max(r['max_orthogonality_error'] for r in rows),
            max_spectral_energy_error=max(r['max_spectral_energy_relative_error'] for r in rows),
            same_model_hash=rows[0]['model_sha256'], allocator=json.loads((root/'runs'/task/'allocator_validation.json').read_text()))
    assert not audit['hash_mismatches']
    summary = {'protocol':str(root/'PROTOCOL.md'), 'tasks':{},
               'uncertainty':'Descriptive mean/sample SD over 5 overlapping calibration draws with a shared fixed B; no significance claims.',
               'metric_note':'Capture is percentage of squared gradient norm at initialization, not downstream accuracy.'}
    spectral_check = None
    if (out/'independent_spectral_seed.json').exists():
        spectral_check = json.loads((out/'independent_spectral_seed.json').read_text())
        check_done = json.loads((root/'independent_spectral_seed_check/DONE.json').read_text())
        assert check_done['complete']
        for name, expected in check_done['artifacts'].items():
            assert sha(root/'independent_spectral_seed_check'/name) == expected
        summary['independent_spectral_seed_sensitivity'] = spectral_check
    capture_rows, score_rows, pair_rows, candidate_rows, rank_rows, variability_rows = [], [], [], [], [], []
    residual_plot = {}
    for task, all_runs in runs.items():
        primary = [r for r in all_runs if r['order'] == 0]
        per_task_scores = []
        for run in all_runs:
            base = {k:run[k] for k in ['task','replicate','order']}
            candidates = []
            for name, m in run['modules'].items():
                for k in range(1,17):
                    row = dict(base, module=name, module_type=module_type(name), shape='x'.join(map(str,m['shape'])),
                        k=k, consistency=m['consistency_raw'][k-1], consistency_used=m['consistency_used'][k-1],
                        recurrence=m['cross_subspace_recurrence'][k-1], b_consistency=m['b_consistency_raw'][k-1],
                        log_cumulative_energy=float(np.log(max(sum(m['mean_block_energy'][:k]),1e-30))),
                        left_fraction=m['left_capture_energy'][k-1]/max(m['b_gradient_energy'],1e-30),
                        right_fraction=m['right_capture_energy'][k-1]/max(m['b_gradient_energy'],1e-30),
                        mean_block_energy=m['mean_block_energy'][k-1])
                    candidates.append(row)
                for method, a in run['allocations'].items():
                    rank_rows.append(dict(base, module=name, module_type=module_type(name), method=method,
                                          rank=a['ranks'][name], cost=m['cost']))
            candidate_rows.extend(candidates)
            for outcome in ['recurrence','b_consistency','left_fraction','right_fraction']:
                for control in ['shape','module_type']:
                    # k=1,2 are useful spectral diagnostics, but allocation choices begin at k=3.
                    for scope, selected in [('all_k',candidates), ('allocatable_k',[r for r in candidates if r['k']>=3])]:
                        x,y = controlled_residuals(selected, outcome, control)
                        s = dict(base, outcome=outcome, control=control, scope=scope,
                                 partial_rank_correlation=corr(x,y))
                        score_rows.append(s)
                        if run['order'] == 0:
                            per_task_scores.append(s)
                        if run['replicate']==0 and run['order']==0 and outcome=='recurrence' and control=='shape' and scope=='allocatable_k':
                            residual_plot[task] = (x,y)
            for method, a in run['allocations'].items():
                capture_rows.append(dict(base, method=method, budget=a['budget_used'],
                    **{k:a[k]*100 for k in ['left_capture','right_capture','left_incremental_capture','right_incremental_capture']},
                    at_r_min_fraction=a['at_r_min_fraction'], at_r_max_fraction=a['at_r_max_fraction']))
            for side in ['left','right']:
                capture_rows.append(dict(base, method='same_shape_shuffle_mean_'+side,
                    **{side+'_capture':100*float(np.mean([s[side+'_capture'] for s in run['shuffles']])),
                       side+'_shuffle_sd':100*float(np.std([s[side+'_capture'] for s in run['shuffles']], ddof=1))}))
        for method in ['energy','consistency','near_uniform']:
            for i,a in enumerate(primary):
                for b in primary[i+1:]:
                    pair_rows.append(dict(task=task, method=method, perturbation='subset',
                        a=a['replicate'], b=b['replicate'], distance=distance(a['allocations'][method]['ranks'],
                        b['allocations'][method]['ranks'],a['modules'])))
            for a in primary:
                b = next(r for r in all_runs if r['replicate']==a['replicate'] and r['order']==1)
                pair_rows.append(dict(task=task, method=method, perturbation='order_block',
                    a=a['replicate'],b=a['replicate'], distance=distance(a['allocations'][method]['ranks'],
                        b['allocations'][method]['ranks'],a['modules'])))
        t = {'lambda':manifest['tasks'][task]['consistency_lambda'], 'capture':{}, 'stability':{}, 'predictivity':{}}
        for method in ['energy','consistency','near_uniform']:
            t['capture'][method] = {side:avg_sd([100*r['allocations'][method][side+'_capture'] for r in primary]) for side in ['left','right']}
            t['capture'][method]['rank_min_fraction'] = avg_sd([r['allocations'][method]['at_r_min_fraction'] for r in primary])
            t['capture'][method]['rank_max_fraction'] = avg_sd([r['allocations'][method]['at_r_max_fraction'] for r in primary])
            t['stability'][method] = {kind:avg_sd([r['distance'] for r in pair_rows if r['task']==task and r['method']==method and r['perturbation']==kind])
                                      for kind in ['subset','order_block']}
        t['paired_capture_gain_pp'] = {side:avg_sd([100*(r['allocations']['consistency'][side+'_capture']-
                                                        r['allocations']['energy'][side+'_capture']) for r in primary]) for side in ['left','right']}
        t['order1_paired_capture_gain_pp'] = {side:avg_sd([100*(r['allocations']['consistency'][side+'_capture']-
                                                        r['allocations']['energy'][side+'_capture']) for r in all_runs if r['order']==1]) for side in ['left','right']}
        t['shuffle_capture'] = {side:avg_sd([100*float(np.mean([s[side+'_capture'] for s in r['shuffles']])) for r in primary]) for side in ['left','right']}
        for outcome in ['recurrence','b_consistency','left_fraction','right_fraction']:
            t['predictivity'][outcome] = {control:avg_sd([s['partial_rank_correlation'] for s in per_task_scores
                if s['outcome']==outcome and s['control']==control and s['scope']=='allocatable_k']) for control in ['shape','module_type']}
        # Exploratory bridge: fixed k=4 reliability vs across-subset module rank variability.
        # Descriptive only; high/low rank saturation can mechanically suppress variability.
        for method in ['energy','consistency']:
            vr = []
            for name,m in primary[0]['modules'].items():
                ranks = [r['allocations'][method]['ranks'][name] for r in primary]
                row = dict(task=task, method=method, module=name, module_type=module_type(name),
                    shape='x'.join(map(str,m['shape'])), k=4,
                    consistency=float(np.mean([r['modules'][name]['consistency_raw'][3] for r in primary])),
                    log_cumulative_energy=float(np.log(max(np.mean([sum(r['modules'][name]['mean_block_energy'][:4]) for r in primary]),1e-30))),
                    rank_sd=float(np.std(ranks,ddof=1)), mean_rank=float(np.mean(ranks)),
                    always_boundary=all(v==2 for v in ranks) or all(v==16 for v in ranks))
                vr.append(row)
            variability_rows.extend(vr)
            t.setdefault('exploratory_score_rank_sd',{})[method] = {}
            for control in ['shape','module_type']:
                x,y = controlled_residuals(vr,'rank_sd',control)
                t['exploratory_score_rank_sd'][method][control] = corr(x,y)
        summary['tasks'][task] = t
    for filename, rows in [('captures.csv',capture_rows),('predictivity.csv',score_rows),('rank_distances.csv',pair_rows),
                           ('candidate_metrics.csv',candidate_rows),('ranks.csv',rank_rows),('exploratory_rank_variability.csv',variability_rows)]:
        write_csv(out/filename,rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    with PdfPages(out/'mechanism_report.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(12,3.9))
        for ax,task in zip(axes,TASKS):
            t=summary['tasks'][task]
            for j,(outcome,label) in enumerate([('recurrence','A/B recurrence'),('b_consistency','B-only consistency'),('right_fraction','Right capture')]):
                for offset,control,marker,color in [(-.10,'shape','o','#247ba0'),(.10,'module_type','s','#d59a36')]:
                    stat=t['predictivity'][outcome][control]
                    ax.errorbar(j+offset,stat['mean'],yerr=stat['sd'],fmt=marker,color=color,capsize=3,
                                label=control.replace('_',' ') if j==0 else None)
            ax.axhline(0,color='.6',lw=.8);ax.set_ylim(-1,1);ax.set_xticks(range(3),['A/B\nrecurrence','B-only\nconsistency','Right\ncapture'])
            ax.set_title(task.upper());ax.set_ylabel('Energy-adjusted rank correlation')
        axes[0].legend(fontsize=8)
        fig.suptitle('Predictivity on independent B: fixed k and shape/type; k = 3..16')
        fig.text(.5,.01,'Mean and sample SD over 5 calibration draws; shared B and overlapping A draws. No significance test.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'01_predictivity.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,kind,title in zip(axes,['subset','order_block'],['Different calibration subsets','Same samples, changed order / blocks']):
            for j,method in enumerate(['energy','consistency']):
                means=[summary['tasks'][t]['stability'][method][kind]['mean'] for t in TASKS]
                ax.bar(np.arange(3)+(j-.5)*.30,means,width=.28,color=COLORS[method],label=LABELS[method])
                for x,task in enumerate(TASKS):
                    vals=summary['tasks'][task]['stability'][method][kind]['values']
                    ax.scatter(x+(j-.5)*.30+np.linspace(-.08,.08,len(vals)),vals,s=12,color='k',alpha=.4)
            ax.set_xticks(range(3),[t.upper() for t in TASKS]);ax.set_ylabel('Cost-weighted rank distance');ax.set_title(title);ax.legend(fontsize=8)
        fig.suptitle('Allocation repeatability at identical parameter budget')
        fig.text(.5,.01,'Subset panel: 10 dependent pairs from 5 draws. Order panel: 5 paired perturbations. Points are descriptive.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(out/'02_repeatability.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,side in zip(axes,['left','right']):
            for j,method in enumerate(['energy','consistency','near_uniform']):
                stats=[summary['tasks'][t]['capture'][method][side] for t in TASKS]
                ax.bar(np.arange(3)+(j-1)*.24,[s['mean'] for s in stats],yerr=[s['sd'] for s in stats],
                       width=.22,color=COLORS[method],label=LABELS[method],capsize=3)
            ax.set_xticks(range(3),[t.upper() for t in TASKS]);ax.set_ylim(0,105);ax.set_ylabel('Captured gradient energy (%)')
            ax.set_title(side.capitalize()+' singular subspace');ax.legend(fontsize=7)
        fig.suptitle('Independent B capture: same A basis for every allocation')
        fig.text(.5,.01,'Budget = 755,712 for all methods. Right space corresponds to LoRA A row space; this is an initialization proxy.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(out/'03_capture.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,side in zip(axes,['left','right']):
            for i,task in enumerate(TASKS):
                stat=summary['tasks'][task]['paired_capture_gain_pp'][side]
                ax.scatter(i+np.linspace(-.1,.1,5),stat['values'],color=COLORS['consistency'],s=30)
                ax.plot([i-.18,i+.18],[stat['mean']]*2,color='black',lw=2)
            ax.axhline(0,color='.6',ls='--');ax.set_xticks(range(3),[t.upper() for t in TASKS]);ax.set_ylabel('Capture gain (percentage points)');ax.set_title(side.capitalize()+' capture: paired differences')
        fig.suptitle('Consistency minus energy-only: every preselected calibration draw')
        fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(out/'04_paired_capture.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,3.7))
        for ax,task in zip(axes,TASKS):
            x,y=residual_plot[task];ax.hexbin(x,y,gridsize=20,mincnt=1,cmap='Blues');ax.set_title(task.upper());ax.set_xlabel('Residual rank of A consistency');ax.set_ylabel('Residual rank of A/B recurrence')
        fig.suptitle('Illustration: first preselected draw; k and shape stratified, energy adjusted')
        fig.text(.5,.01,'Nested module/rank candidates are not independent replicates. Main results above use all 5 draws.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(out/'05_controlled_relationship.png',dpi=180);pdf.savefig(fig);plt.close(fig)
    lines=['# 独立校准机制验证结果','',f'实验目录：`{root}`','',
        '已完成3任务×（1个独立B + 5个A×2种顺序）=33次校准。没有做下游训练；以下都是初始化点的机制指标。',
        'A/B均来自训练集，彼此在样本ID及完全相同文本层面隔离。每次512条、4块；同预算755,712。分类头固定seed42。',
        '主要结果用每组预先指定的order0；均值±样本标准差来自5个A重复。A之间有交集，B共用，不能当作5次完全独立实验。','',
        '|任务|λ|A一致性→A/B重现：控制k/shape/能量|进一步控制模块类型|独立B自身一致性相关|',
        '|---|---:|---:|---:|---:|']
    def fmt(s,d=3): return f'{s["mean"]:.{d}f} ± {s["sd"]:.{d}f}'
    for task in TASKS:
        t=summary['tasks'][task];p=t['predictivity']
        lines.append(f'|{task.upper()}|{t["lambda"]}|{fmt(p["recurrence"]["shape"])}|{fmt(p["recurrence"]["module_type"])}|{fmt(p["b_consistency"]["shape"])}|')
    lines+=['','相关统计只用可分配增量k=3…16。在每个相同k、形状/类型组内做秩转换，并将log累计块谱能量作为控制变量残差化。无p值或显著性宣称。','',
        '|任务|跨子集距离 λ=0|跨子集距离 完整方法|换序距离 λ=0|换序距离 完整方法|',
        '|---|---:|---:|---:|---:|']
    for task in TASKS:
        s=summary['tasks'][task]['stability'];lines.append(f'|{task.upper()}|{s["energy"]["subset"]["mean"]:.4f}|{s["consistency"]["subset"]["mean"]:.4f}|{s["energy"]["order_block"]["mean"]:.4f}|{s["consistency"]["order_block"]["mean"]:.4f}|')
    lines+=['','距离为Σ成本×|rank差|/(2P)，越小越稳定；跨子集为10个相关配对的描述均值，换序为5组配对。Near-uniform固定表距离自然为0，不能据此推断其更优。','',
        '|任务|左捕获 λ=0 (%)|左捕获 完整 (%)|配对增量 (pp)|右捕获 λ=0 (%)|右捕获 完整 (%)|配对增量 (pp)|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        t=summary['tasks'][task];c=t['capture'];d=t['paired_capture_gain_pp']
        lines.append(f'|{task.upper()}|{fmt(c["energy"]["left"])}|{fmt(c["consistency"]["left"])}|{fmt(d["left"])}|{fmt(c["energy"]["right"])}|{fmt(c["consistency"]["right"])}|{fmt(d["right"])}|')
    lines+=['','每个A共用同一SVD基底，仅rank表改变。左/右分别是Σ||UᵀG_B||²和Σ||G_BV||²除以Σ||G_B||²；右侧对应实际svd_a_zero_b的A行空间。',
        '投影基底使用gesvd提高数值正交精度；分配调用冻结原算法，首组两个λ均与原完整路径rank表完全相同。','',
        '|任务|同预算近均匀：左/右 (%)|完整rank同形状置换100次：左/右均值 (%)|A子集交集均值/512|',
        '|---|---:|---:|---:|']
    for task in TASKS:
        t=summary['tasks'][task];u=t['capture']['near_uniform'];s=t['shuffle_capture'];ov=np.asarray(manifest['tasks'][task]['a_pair_overlap_counts'])
        lines.append(f'|{task.upper()}|{u["left"]["mean"]:.3f} / {u["right"]["mean"]:.3f}|{s["left"]["mean"]:.3f} / {s["right"]["mean"]:.3f}|{ov[np.triu_indices(5,1)].mean():.1f}|')
    lines+=['','另一种顺序的配对捕获差（完整方法−λ=0，pp），用于检查主结果对顺序的依赖：','',
        '|任务|左侧增量 order1|右侧增量 order1|','|---|---:|---:|']
    for task in TASKS:
        d=summary['tasks'][task]['order1_paired_capture_gain_pp']
        lines.append(f'|{task.upper()}|{fmt(d["left"])}|{fmt(d["right"])}|')
    lines+=['','探索性关系：固定k=4的一致性 vs 模块rank跨A子集标准差。负相关表示高一致性模块的rank波动较小，正相关表示相反。控制形状和谱能量，仍受rank上下界影响，不能解释为因果。','',
        '|任务|λ=0分配的相关|完整方法分配的相关|','|---|---:|---:|']
    for task in TASKS:
        v=summary['tasks'][task]['exploratory_score_rank_sd']
        lines.append(f'|{task.upper()}|{v["energy"]["shape"]:.3f}|{v["consistency"]["shape"]:.3f}|')
    if spectral_check:
        lines+=['','独立随机谱种子复核（主结果完成后的数值敏感性检查，另外3次B校准）：将B的谱随机种子由3407改为103407，样本、模型、dropout种子和A全部保持原样。','',
            '|任务|原种子相关|B独立谱种子相关|B梯度最大相对差|','|---|---:|---:|---:|']
        for task in TASKS:
            s=spectral_check[task]['predictivity']['recurrence']['shape']
            lines.append(f'|{task.upper()}|{s["original_seed_mean"]:.4f}|{fmt(s,4)}|{spectral_check[task]["max_gradient_relative_difference"]:.2e}|')
        lines+=['','主实验33次，加此项复核3次，共36次校准；这项复核没有改变原始主结果或选择新的lambda。']
    lines+=['','解释边界：','',
        '- 评分的独立预测性、分配重复性、独立梯度捕获是三个不同命题，应按各任务实际结果分别表述。',
        '- 校准使用原实现的train模式、普通CE；改变顺序同时改变样本对应的dropout掩码，换序结果不隔离纯分块效应。',
        '- 只有一个分类头初始化与一个固定B。主表已有训练种子实验不能替代本实验的独立B重抽样验证。',
        '- 右侧捕获是初始化处的局部代理；不能推出最终准确率更高、性能更稳或能力保留率。',
        '- 原始与截断一致性、逐模块逐rank指标、全部rank和100次置换均已保存；负向结果也保留。',
        '- exploratory_rank_variability.csv额外给出固定k=4一致性与模块rank标准差的关系，只属探索性诊断，受rank边界饱和影响。','',
        '文件：`mechanism_report.pdf`（5页图）、`summary.json`、`captures.csv`、`predictivity.csv`、`rank_distances.csv`、`candidate_metrics.csv`、`ranks.csv`、`validation.json`。']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(__file__,out/Path(__file__).name)
    import fitz
    document = fitz.open(out/'mechanism_report.pdf')
    outside = []
    for page in document:
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if min(x0,y0)<-1 or x1>page.rect.width+1 or y1>page.rect.height+1:
                        outside.append({'page':page.number+1,'text':span['text'],'bbox':span['bbox']})
    assert not outside, outside
    audit.update(pdf_pages=len(document), pdf_text_outside_page=outside,
        source_manifest_sha256=sha(root/'manifest.json'), analysis_sha256=sha(__file__),
        report_pdf_sha256=sha(out/'mechanism_report.pdf'), total_calibrations=33, primary_draws_per_task=5,
        candidate_rows=len(candidate_rows), rank_rows=len(rank_rows), csv_rows={
            'captures':len(capture_rows),'predictivity':len(score_rows),'distances':len(pair_rows)})
    audit['additional_spectral_seed_calibrations'] = 3 if spectral_check else 0
    audit['all_calibrations_including_sensitivity'] = 36 if spectral_check else 33
    (out/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/workspace/SpecCon-LoRA/experiments/independent_calibration_mechanism_20261019_v1'))
    p.add_argument('--out',type=Path,default=Path('/workspace/SpecCon-LoRA/reports/independent_calibration_mechanism_20261019'))
    args=p.parse_args();main(args.root,args.out)
