#!/usr/bin/env python3
"""Build a categorized, auditable results bundle without modifying source runs."""
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
from zoneinfo import ZoneInfo
import zipfile

ROOT=Path('/workspace/SpecCon-LoRA')
DEST=ROOT/'reports/all_ablation_summary_20261020'
OLD=ROOT/'reports/acl_experiment_tables_20261016'
MECH=ROOT/'reports/independent_calibration_mechanism_20261019'
MEXP=ROOT/'experiments/independent_calibration_mechanism_20261019_v1'
DIRS=['00_总览','01_组件与初始化','02_lambda敏感性','03_RDrop消融','04_rank置换',
      '05_独立校准机制','06_历史调参与补充对照','07_论文表格与PDF','08_复现与溯源','09_证据缺口与排除项']
SEEDS=[42,777,2026]
TASKS=['cola','mrpc','rte']
PROVENANCE=[]


def log(s): print(s,flush=True)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def copy(src,dst):
    src,dst=Path(src),Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    PROVENANCE.append(dict(original_path=str(src),archive_path=str(dst.relative_to(DEST)),
                           bytes=dst.stat().st_size,sha256=sha(dst)))


def csvout(path,rows):
    if not rows:return
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def avg(xs):return dict(mean=st.mean(xs),sd=st.stdev(xs) if len(xs)>1 else None,n=len(xs),values=xs)
def fmt(s,d=3):return f'{s["mean"]:.{d}f} ± {s["sd"]:.{d}f}' if s.get('sd') is not None else f'{s["mean"]:.{d}f}'


def refresh_tables():
    paper=DEST/DIRS[7]/'ACL表格源码'
    paper.mkdir(parents=True,exist_ok=True)
    for name in ['collect_results.py','build_tables.py','acl.sty','acl_natbib.bst','tool_sources.json']:
        copy(OLD/name,paper/name)
    p=paper/'collect_results.py'
    text=p.read_text()
    old="sensitivity.append({'task': task, 'lambda': 0, 'results': [zeros[task]] if zeros[task] else []})"
    new="sensitivity.append({'task': task, 'lambda': 0, 'results': [x['result'] for x in components if x['task'] == 'rte' and x['variant'] == 'D' and x['result'] is not None]})"
    assert old in text
    p.write_text(text.replace(old,new))
    with open(paper/'collection.log','w') as out:
        subprocess.run([sys.executable,str(p)],cwd=paper,stdout=out,stderr=subprocess.STDOUT,check=True)
    p=paper/'build_tables.py';text=p.read_text()
    text=text.replace('RTE λ=0 has only the completed seed-42 ablation. ',
                      'RTE λ=0 now includes all three completed component-D seeds. ')
    text=text.replace('QNLI λ=0/0.7/1.5 and RTE λ=0.5/1.5 each have all three completed seeds.',
                      'QNLI λ=0/0.7/1.5 and RTE λ=0/0.5/1.5 each have all three completed seeds.')
    p.write_text(text)
    spec=importlib.util.spec_from_file_location('bundle_tables',p)
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    b.main_results();b.components();b.rdrop();b.lambda_tables();b.historical();b.appendices();b.completed_extensions()
    m=json.loads((MECH/'summary.json').read_text())
    contrasts=[]
    for label,a,c in [('F−D','F','D'),('E−C','E','C'),('B−A','B','A'),('D−C','D','C'),('F−E','F','E')]:
        contrasts.append([label]+[fmt(avg([b.component(t,a,s)['score']-b.component(t,c,s)['score'] for s in SEEDS])) for t in TASKS])
    b.add('16_paired_contrasts_multiseed','同预算组件差值（三种子）',['Contrast','CoLA','MRPC','RTE'],contrasts,
          'Paired final-epoch differences, mean and sample SD over seeds 42/777/2026. Every listed contrast preserves '
          'its parameter budget. F-D and E-C test consistency with SVD and random initialization; B-A, D-C and F-E '
          'test the whole SVD initialization scheme within each allocation family. Three seeds do not establish significance.',align='lrrr')
    rows=[]
    for task in TASKS:
        t=m['tasks'][task];p=t['predictivity'];rows.append([task.upper(),str(t['lambda']),fmt(p['recurrence']['shape']),
            fmt(p['recurrence']['module_type']),fmt(p['b_consistency']['shape'])])
    b.add('17_independent_score_predictivity','独立校准：评分预测性',
          ['Task','λ','A/B: shape control','A/B: type control','B-only consistency'],rows,
          'Independent train-only calibration analysis: five A draws of 512 samples, four blocks per draw, and a fixed disjoint '
          'B of 512 samples. Rank correlations are stratified by k and shape/type and adjusted for spectral energy; k=3..16. '
          'Mean and sample SD are descriptive: A draws overlap and share B. No downstream training is performed.',align='lrrrr')
    rows=[]
    for task in TASKS:
        t=m['tasks'][task];s=t['stability'];rows.append([task.upper(),f'{s["energy"]["subset"]["mean"]:.4f}',
            f'{s["consistency"]["subset"]["mean"]:.4f}',fmt(t['paired_capture_gain_pp']['left']),fmt(t['paired_capture_gain_pp']['right'])])
    b.add('18_independent_stability_capture','独立校准：分配重复性与梯度捕获',
          ['Task','Distance: λ=0','Distance: full','Left capture Δ (pp)','Right capture Δ (pp)'],rows,
          'Cost-weighted rank distance is lower for more repeatable allocations. Mean distances summarize the ten dependent '
          'pairs of five calibration draws. Capture differences are full minus energy-only allocation, using the same A SVD '
          'bases and the independent B gradient at initialization. Every allocation uses exactly 755,712 parameters. '
          'Capture is a local gradient proxy, not accuracy or capability retention. The right space corresponds to LoRA A.',align='lrrrr')
    status=next(t for t in b.TABLES if t['id']=='11_completion_status') if any(t['id']=='11_completion_status' for t in b.TABLES) else next(t for t in b.TABLES if t['title']=='实验完成情况')
    status['rows'] += [['Independent calibration mechanism','3 tasks × (1 B + 5 A × 2 orders)','33/33','Complete'],
                       ['Independent spectral-seed check','3 heldout B recalibrations','3/3','Complete']]
    status['caption']=status['caption'].replace('Planned mechanism or budget studies without outputs are not presented as findings.',
        'Independent calibration is now complete. Equal-budget downstream training curves remain outside this completed set.')
    b.export()
    with open(paper/'all_tables.tex','a') as f:
        for t in b.TABLES[15:]:f.write('\\input{tables/'+t['id']+'.tex}\n\\clearpage\n')
    with open(paper/'compile.log','w') as out:
        subprocess.run(['/workspace/.tools/tectonic-0.17.0/tectonic','--keep-logs','acl_tables.tex'],cwd=paper,
                       stdout=out,stderr=subprocess.STDOUT,check=True)
    return paper,b,m


def collect_stats(b,m):
    d=b.D
    result={'created':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'components':{},'contrasts':{},'rdrop':{},'rank_shuffle':{},'lambda':[], 'mechanism':m['tasks']}
    component_rows=[];contrast_rows=[];rdrop_rows=[];shuffle_rows=[];lambda_rows=[]
    for t in TASKS:
        result['components'][t]={v:avg([b.component(t,v,s)['score'] for s in SEEDS]) for v in 'ABCDEF'}
        for v in 'ABCDEF':
            for s in SEEDS:
                r=b.component(t,v,s)
                component_rows.append(dict(task=t,variant=v,seed=s,score=r['score'],parameters=r['parameters'],metrics=r['metrics']))
        result['contrasts'][t]={}
        for name,a,c in [('F-D','F','D'),('E-C','E','C'),('B-A','B','A'),('D-C','D','C'),('F-E','F','E')]:
            values=[b.component(t,a,s)['score']-b.component(t,c,s)['score'] for s in SEEDS]
            result['contrasts'][t][name]=avg(values)
            contrast_rows.append(dict(task=t,contrast=name,seed42=values[0],seed777=values[1],seed2026=values[2],
                                      mean=st.mean(values),sample_sd=st.stdev(values)))
        result['rdrop'][t]={}
        for method in ['uniform','speccon']:
            result['rdrop'][t][method]={}
            by_alpha={}
            for alpha in [0,1]:
                rr=[x['result'] for x in d['extended_rdrop'] if x['task']==t and x['method']==method and x['alpha']==alpha]
                assert sorted(x['seed'] for x in rr)==SEEDS
                rr=sorted(rr,key=lambda x:SEEDS.index(x['seed']));by_alpha[alpha]=rr
                result['rdrop'][t][method][str(alpha)]=avg([x['score'] for x in rr])
                for r in rr:rdrop_rows.append(dict(task=t,method=method,alpha=alpha,seed=r['seed'],score=r['score'],parameters=r['parameters'],metrics=r['metrics']))
            result['rdrop'][t][method]['gain']=avg([a['score']-c['score'] for a,c in zip(by_alpha[1],by_alpha[0])])
    for t in ['mrpc','rte']:
        orig=[b.component(t,'F',s) for s in SEEDS]
        shuf=[next(x['result'] for x in d['rank_shuffle'] if x['task']==t and x['seed']==s) for s in SEEDS]
        result['rank_shuffle'][t]=dict(original=avg([x['score'] for x in orig]),permuted=avg([x['score'] for x in shuf]),
            gain=avg([a['score']-c['score'] for a,c in zip(orig,shuf)]))
        for kind,rr in [('original',orig),('permuted',shuf)]:
            for r in rr:shuffle_rows.append(dict(task=t,placement=kind,seed=r['seed'],score=r['score'],parameters=r['parameters'],metrics=r['metrics']))
    # Complete same-recipe component D/F values extend lambda coverage for the small tasks.
    groups={(x['task'],float(x['lambda'])):x['results'] for x in d['sensitivity']}
    for t in TASKS:
        for variant in ['D','F']:
            rr=[b.component(t,variant,s) for s in SEEDS]
            groups[t,float(rr[0]['hyperparameters']['lambda'])]=rr
    for t in ['sst2','stsb','mnli','qqp']:
        lam=float(d['selected'][t][0]['hyperparameters']['lambda'])
        groups[t,lam]=d['selected'][t]
        groups[t,0.0]=[d['lambda0'][t]]
    for (t,lam),rr in sorted(groups.items()):
        assert all(r['hyperparameters']['lambda']==lam for r in rr)
        result['lambda'].append(dict(task=t,lambda_value=lam,**avg([r['score'] for r in rr])))
        for r in rr:lambda_rows.append(dict(task=t,lambda_value=lam,seed=r['seed'],score=r['score'],parameters=r['parameters'],
            epochs=r['epochs'],metrics=r['metrics']))
    for directory,filename,rows in [(DIRS[1],'逐种子结果.csv',component_rows),(DIRS[1],'配对差值_三种子.csv',contrast_rows),
        (DIRS[2],'lambda_全部可核验逐种子.csv',lambda_rows),(DIRS[3],'RDrop_36格逐种子.csv',rdrop_rows),(DIRS[4],'rank置换_逐种子.csv',shuffle_rows)]:
        csvout(DEST/directory/filename,rows)
    dump(DEST/DIRS[0]/'统计汇总_未舍入.json',result)
    return result


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def write_summaries(r,paper):
    labels={'cola':'CoLA','mrpc':'MRPC','rte':'RTE'}
    definitions=[['A','均匀 rank=8','随机 A、零 B','1,327,104'],['B','均匀 rank=8','梯度 SVD A、零 B','1,327,104'],
        ['C','分块谱，λ=0','随机 A、零 B','755,712'],['D','分块谱，λ=0','梯度 SVD A、零 B','755,712'],
        ['E','分块谱＋一致性','随机 A、零 B','755,712'],['F','分块谱＋一致性','梯度 SVD A、零 B','755,712']]
    component_table=md_table(['方案','CoLA MCC×100','MRPC Accuracy×100','RTE Accuracy×100'],
        [[v]+[fmt(r['components'][t][v]) for t in TASKS] for v in 'ABCDEF'])
    contrast_table=md_table(['同预算配对差值','CoLA','MRPC','RTE'],
        [[name]+[fmt(r['contrasts'][t][name]) for t in TASKS] for name in ['F-D','E-C','B-A','D-C','F-E']])
    text='# 组件与初始化消融\n\n完成54/54格：3任务×6方案×3种子。\n\n'+md_table(['ID','分配','初始化','适配器参数'],definitions)+'\n\n'+component_table+'\n\n'+contrast_table+'''

主要结论：完整方法F三任务均值均高于同预算的D；F-D分别为+1.632、+1.716、+1.805个指标点。但逐种子并非全胜，CoLA和RTE各有一个负差值。随机初始化下E-C并不普遍为正，CoLA三种子均为负；一致性效果依赖初始化方式。

F-E为+2.186、+1.307、+1.324，支持在完整分配下使用整套SVD初始化方案。不能把这个对照单独解释成“方向有效”，因为初始化的方向与尺度没有匹配。

A/B预算1,327,104，C–F预算755,712。F比A少43.06%的适配器参数，但跨预算差值不能被描述为等预算纯分配收益。所有组都使用R-Drop α=1；MRPC固定label smoothing=0.05；种子42/777/2026，固定末轮。
'''
    (DEST/DIRS[1]/'README_结论.md').write_text(text)
    lambda_table=md_table(['任务','λ','n','均值±样本标准差'],[[x['task'].upper(),x['lambda_value'],x['n'],fmt(x,4)] for x in r['lambda']])
    (DEST/DIRS[2]/'README_结论.md').write_text('# λ敏感性与一致性消融\n\n'+lambda_table+'''

λ是本方法的子空间一致性指数，不是dropout概率，也不是R-Drop的α。λ=0对照仍使用相同的4块谱，仅去掉一致性权重。

RTE：λ=0/0.5/1.5三种子均值分别为78.7004/80.5054/79.4224，当前保留0.5。
QNLI：λ=0/0.7/1.5分别为93.0319/92.9099/92.8489。按用户要求，主表保留非零λ=0.7，λ=0只作消融；不能声称非零λ在该任务均值上更优。

CoLA/MRPC/RTE的λ=0三种子已由组件D补齐。八任务seed42对照中，MNLI、QQP、QNLI λ=1.5略降，STS-B基本不变；不能表述为所有任务均受益。跨任务采用不同λ，属于已有任务配置，不是统一λ实验。

本表中的n是已留存、可核验结果数量。历史清单中的其他候选值不自动构成可用结果；缺失记录见09目录。
''')
    rows=[]
    for method in ['uniform','speccon']:
        for alpha in ['0','1']:rows.append([method,alpha]+[fmt(r['rdrop'][t][method][alpha]) for t in TASKS])
    rdtable=md_table(['方法','α','CoLA','MRPC','RTE'],rows)
    (DEST/DIRS[3]/'README_结论.md').write_text('# R-Drop配方消融\n\n完成36/36格：3任务×2方法×开关×3种子。\n\n'+rdtable+'''

SpecCon开启R-Drop后的三种子均值增益：CoLA +1.050、MRPC +1.634、RTE +1.203。RTE的配对增益为[+3.610, 0, 0]，主要来自seed42，不能说每个种子都依赖R-Drop。

Uniform LoRA的增益：CoLA +1.500、MRPC −0.163、RTE +0.602，说明正则化配方的作用与方法/任务有关。

α=0使用一次前向，α=1使用两次前向及对称KL；该对照比较整套配方，不是相同计算量下仅移除KL。MRPC所有格保留标签平滑0.05。方法内预算相同，Uniform与SpecCon之间预算不同。
''')
    shuffle_table=md_table(['任务','原始分配','同形状置换','原始−置换'],[[labels[t],fmt(r['rank_shuffle'][t]['original']),
        fmt(r['rank_shuffle'][t]['permuted']),fmt(r['rank_shuffle'][t]['gain'])] for t in ['mrpc','rte']])
    (DEST/DIRS[4]/'README_结论.md').write_text('# rank放置位置消融\n\n完成6次置换训练，并配对复用6个原始F结果。\n\n'+shuffle_table+'''

MRPC原始分配平均高0.735点，三个种子中两个为正；RTE平均高2.407点，三个种子全部为正。这支持rank放置位置有价值，RTE证据更一致。

严格在相同矩阵形状内置换，保持每个形状组的rank多重集合、实际预算755,712和训练配方。每个模块从自己的梯度重新生成SVD初始化。训练种子42/777/2026分别配对置换种子101/202/303；没有搜索最佳置换，也不是训练种子×置换种子的完全交叉实验。
''')
    mechanism_table=md_table(['任务','评分→独立重现相关','rank距离 λ=0→完整','捕获差 左/右(pp)'],
        [[labels[t],f'{r["mechanism"][t]["predictivity"]["recurrence"]["shape"]["mean"]:.3f}',
          f'{r["mechanism"][t]["stability"]["energy"]["subset"]["mean"]:.4f} → {r["mechanism"][t]["stability"]["consistency"]["subset"]["mean"]:.4f}',
          f'{r["mechanism"][t]["paired_capture_gain_pp"]["left"]["mean"]:+.4f} / {r["mechanism"][t]["paired_capture_gain_pp"]["right"]["mean"]:+.4f}'] for t in TASKS])
    coverage=[['组件×初始化','54/54结果格','3任务×6方案×3种子'],['R-Drop配方','36/36结果格','其中部分复用组件A/F'],
        ['八任务λ=0对照','8/8','seed42；与组件D等部分重叠'],['RTE/QNLI λ=1.5','6/6','两任务各3种子'],
        ['QNLI λ=0.7','3/3','主表采用此非零λ'],['rank置换','6/6新训练','另复用6个原始分配'],
        ['独立校准机制','33/33校准','每任务1个B＋5个A×2种顺序'],['独立谱种子复核','3/3校准','主实验后的数值敏感性检查']]
    summary='''# SpecCon-LoRA 消融实验总总结

统计日期：2026-10-20。范围为当前项目中留存且可核验的正式消融、独立机制验证及历史探索记录。

**总体判断：下游组件实验支持完整方法中的一致性与SVD初始化组合，rank置换支持分配位置有价值。机制实验支持评分的独立预测性，但不支持“跨子集rank分配更稳定”的普遍说法。**

## 完成情况与阅读口径

'''+md_table(['实验','完成情况','说明'],coverage)+'''

以上是结果格/校准次数，存在复用，不能相加当作独立训练次数。三种子均为42/777/2026，固定末轮；CoLA为MCC×100、MRPC/RTE为Accuracy×100；标准差使用ddof=1。机制实验的5次校准抽样不是5个训练种子。

## 1. 组件与初始化

'''+md_table(['ID','分配','初始化','参数量'],definitions)+'\n\n'+component_table+'\n\n'+contrast_table+'''

F-D的三任务均值均为正（+1.632/+1.716/+1.805），但不能声称每个种子都提升。E-C在CoLA为负、MRPC接近零、RTE为正，说明一致性的收益依赖初始化。F-E三个任务均值为正，支持完整分配下的SVD初始化方案。

A/B与C–F预算不同：1,327,104 vs 755,712。完整方法参数少43.06%，这不等价于已经证明等预算下的分配优势。初始化对照同时改变方向与尺度，不能只归因于方向。

## 2. λ敏感性

RTE λ=0/0.5/1.5三种子均值为78.7004/80.5054/79.4224；当前保留0.5。旧PDF仅列RTE λ=0的seed42，本次已用组件D修正为3个种子。

QNLI λ=0/0.7/1.5均值为93.0319/92.9099/92.8489。主表按用户要求保留λ=0.7；λ=0只作为消融。λ=0.7优于1.5均值但低于0，不能写“一致性在QNLI均值上更优”。其余任务及逐种子数据见02目录。

λ属于本方法的一致性项；LoRA dropout是丢弃概率，R-Drop α是训练正则化强度，三者不同。

## 3. R-Drop配方

'''+rdtable+'''

SpecCon开R-Drop的均值增益为+1.050/+1.634/+1.203。RTE的增益仅来自seed42，另两个种子为0。α=0/1分别使用1/2次前向，因此这是整套配方消融，不是等计算量纯KL对照。

## 4. rank放置位置

'''+shuffle_table+'''

同形状置换保持rank集合及755,712预算；MRPC三种子有两个正差，RTE三个均为正。RTE更明确地支持“容量放在哪些模块”有价值。

## 5. 独立校准机制

'''+mechanism_table+'''

评分相关性已控制rank、矩阵形状及谱能量，进一步按模块类型控制仍较强；独立B谱随机种子复核结论保持。距离越小越稳定，三个任务的跨子集距离均变大。捕获增益主要见RTE左子空间，右子空间增量很小。

每任务5个A子集与一个固定B；A/B隔离，A重复之间允许重叠。没有进行下游训练，指标是初始化处的局部梯度代理，不能解读成准确率或能力保留率。

## 6. 历史探索、背景对照与尚缺证据

历史学习率、γ、λ、dropout、标签平滑等配置搜索单独存入06目录。部分仅剩配置清单或选中结果，不重建已删除的分数，不将保留最佳值当成完整扫描。

GLUE主表、DPLoRA本地/论文参考、数学代码和MT-Bench结果作为背景另列，不能混充严格消融：训练配方、预算、种子和评价协议可能不同。MT-Bench和数学代码历史记录尤其不能当成多种子公平因果对照。

主要证据缺口：等预算下游训练与预算曲线、完整H/校准规模/γ单因素敏感性、初始化范数匹配、独立校准子集对应的成对下游训练、跨模型机制验证。历史清单不等于这些已完成。

## 建议论文表述

> 组件消融显示，一致性加权与梯度SVD初始化的组合在所考察的小型GLUE任务上取得更高的三种子均值；rank置换实验支持模块间容量放置的重要性。独立校准实验进一步表明，一致性评分能够预测可复现的梯度子空间结构，但该预测性没有普遍转化为更稳定的秩分配，梯度捕获增益具有任务和投影方向依赖性。

所有正负结果均保留。三种子均值和本次描述性相关不自动意味着统计显著。
'''
    (DEST/DIRS[0]/'全部消融总总结.md').write_text(summary)
    csvout(DEST/DIRS[0]/'完成情况.csv',[dict(experiment=a,completed=b,note=c) for a,b,c in coverage])
    (DEST/DIRS[9]/'README_缺口与排除项.md').write_text('''# 证据缺口与排除规则

1. 正式组件版本为glue_main_ablation_3seeds_gpu12_20261016_v3；v1/v2预检失败，里面的引用副本不计新增结果。
2. 正式seed42 R-Drop版本为v2；v1预检失败，不计为消融结果。
3. preflight/smoke、未完成末轮、失败版本、引用副本都不能增加训练重复数。
4. 原始选中报告中的QNLI λ=0作为历史记录保留；本包主表明确采用λ=0.7。
5. 历史搜索仅有manifest而没有完整metrics.json的配置，列为“当前未留存可核验指标”，不等同于从未运行，也不能补造分数。
6. 尚无等预算的完整下游训练曲线；机制中的同预算近均匀投影对照不能替代它。
7. 历史γ/学习率搜索存在配置清单，但完整留存的控制变量结果不足以形成可靠完整敏感性曲线。H、校准规模、初始化范数匹配与跨模型机制证据也尚不齐。
8. 本轮机制验证不支持更稳定rank的普遍论断；该负向结果必须与正向评分预测性同时呈现。

历史版本status副本、全目录扫描记录及实际来源请见08目录。原模型权重、全量梯度缓存保持在原工作区，包内提供位置、大小和已记录哈希索引；本包用于阅读结果、审计和复现设置，不是模型/数据集完整镜像。
''')
    return summary


def main():
    if DEST.exists():raise RuntimeError(f'Output already exists: {DEST}')
    for directory in DIRS:(DEST/directory).mkdir(parents=True)
    log('Refreshing source metrics and correcting RTE lambda=0 coverage...')
    paper,b,m=refresh_tables()
    log('Computing paired, per-seed statistics...')
    r=collect_stats(b,m)
    summary=write_summaries(r,paper)
    dump(DEST/'BUILD_STATE.json',dict(paper=str(paper),stage='summaries_ready'))
    log(str(DEST))


if __name__=='__main__':main()
