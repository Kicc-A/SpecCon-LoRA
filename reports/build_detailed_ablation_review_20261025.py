"""Build a traceable Chinese ablation review, editable tables and publication figures."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import html
import json
import math
import re
import shutil
import statistics as st
import tempfile
import unicodedata
import xml.etree.ElementTree as ET

import fitz
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pypdf
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from fontTools import subset
from fontTools.ttLib import TTFont
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path('/workspace/SpecCon-LoRA')
OUT = ROOT / 'reports/ablation_detailed_review_20261025'
OUT.mkdir(exist_ok=True)
ASSETS = OUT / 'assets'
ASSETS.mkdir(exist_ok=True)
SOURCES = OUT / 'sources'
SOURCES.mkdir(exist_ok=True)
STEM = 'SpecCon-LoRA_消融实验详解与论文修改参考'
EXT = ROOT / 'experiments/extended_ablation_20261021_v2'
OLD = ROOT / 'reports/all_ablation_summary_20261020'
TASKS = ['cola', 'mrpc', 'rte']
LABEL = {'cola':'CoLA', 'mrpc':'MRPC', 'rte':'RTE'}
SEEDS = [42,777,2026]
STAMP = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
source_records=[]

def capture(path, alias):
    path=Path(path)
    data=path.read_bytes()
    (SOURCES/alias).write_bytes(data)
    source_records.append(dict(id=alias,path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    return data

def js(path, alias): return json.loads(capture(path,alias))
def rows(path, alias): return list(csv.DictReader(capture(path,alias).decode('utf-8-sig').splitlines()))

R=js(EXT/'results.json','extended_results.json')
P=js(EXT/'analysis/paired_results.json','extended_paired_results.json')
assert hashlib.sha256((SOURCES/'extended_results.json').read_bytes()).hexdigest()==P['results_sha256']
assert R['completed']==R['total']==len(R['runs'])==219
M=js(ROOT/'reproduction/current_glue_manifest.json','current_glue_manifest.json')
COMP=rows(OLD/'01_组件与初始化/逐种子结果.csv','components.csv')
RD=rows(OLD/'03_RDrop消融/RDrop_36格逐种子.csv','rdrop.csv')
LAM=rows(OLD/'02_lambda敏感性/lambda_全部可核验逐种子.csv','lambda.csv')
SH=rows(OLD/'04_rank置换/rank置换_逐种子.csv','rank_permutation.csv')
HPO_ROOT=ROOT/'experiments/qnli_qqp_hparam_sweep_20261022_v1'
HPO=js(HPO_ROOT/'status.json','hpo_status.json')
HPO_REPORT=capture(HPO_ROOT/'analysis/REPORT.md','hpo_report.md').decode()
HPO_CANDIDATES=rows(HPO_ROOT/'analysis/candidates.csv','hpo_candidates.csv')
capture(EXT/'PROTOCOL.md','extended_protocol.md')
capture(ROOT/'reports/independent_calibration_mechanism_20261019/RESULTS_zh.md','mechanism_results.md')
capture(ROOT/'reports/independent_calibration_mechanism_20261019/INTERPRETATION_zh.md','mechanism_interpretation.md')
capture(ROOT/'README.md','method_readme.md')
capture(OLD/'04_rank置换/README_结论.md','rank_permutation_protocol.md')
capture(ROOT/'reproduction/extended_ablation_support.py','extended_ablation_support.py')

allrows=R['runs']+R['references']
lookup={(r['task'],r['name'],int(r['seed'])):float(r['score']) for r in allrows}
summaries={(r['task'],r['name']):r for r in P['summaries']}
contrasts={(r['task'],r['comparison']):r for r in P['contrasts']}
comp={(r['task'],r['variant'],int(r['seed'])):float(r['score']) for r in COMP}
assert len(comp)==54

def describe(v): return (st.mean(v),st.stdev(v) if len(v)>1 else None)
def fmt(v, digits=3):
    m,s=describe(v)
    return f'{m:.{digits}f} ± {s:.{digits}f}' if s is not None else f'{m:.{digits}f}（n=1）'
def vf(task,name):return [lookup[task,name,s] for s in SEEDS]
def cf(task,name):return [comp[task,name,s] for s in SEEDS]
def vals(task,name):return fmt(vf(task,name))
def delta(v):
    m,s=describe(v)
    return f'{m:+.3f} ± {s:.3f}'
def wins(v):return '/'.join(str(sum(1 for x in v if f(x))) for f in [lambda x:x>1e-8,lambda x:abs(x)<=1e-8,lambda x:x< -1e-8])
def contrast(task,key):return [x['difference'] for x in contrasts[task,key]['paired_values']]
def table(headers,data):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in data])
def summary_table(names):
    return table(['配置','CoLA','MRPC','RTE'],[[lab]+[vals(t,name) for t in TASKS] for name,lab in names])
def contrast_table(names):
    return table(['配对差值','CoLA','MRPC','RTE'],[[lab]+[delta(contrast(t,key)) for t in TASKS] for key,lab in names])
def seed_delta_table(key):
    return table(['任务','seed42','seed777','seed2026','均值 ± SD','胜/平/负'],[[LABEL[t]]+[f'{x:+.3f}' for x in contrast(t,key)]+[delta(contrast(t,key)),wins(contrast(t,key))] for t in TASKS])

# Recompute stored summaries and all ordinary paired contrasts from run-level data.
for (task,name),r in summaries.items():
    v=[float(x['score']) for x in allrows if x['task']==task and x['name']==name]
    assert len(v)==r['n'] and abs(st.mean(v)-r['mean'])<1e-9
    if len(v)>1:assert abs(st.stdev(v)-r['sample_std'])<1e-9
for r in P['contrasts']:
    if r['unit']=='training_seed' and r['n']:
        d=[lookup[r['task'],r['left'],s]-lookup[r['task'],r['right'],s] for s in SEEDS]
        assert abs(st.mean(d)-r['mean'])<1e-9

pages=[]
def page(title,body,compact=False):pages.append(dict(title=title,body=body.strip(),compact=compact))

page('01 结论总览与阅读路径',f'''
# SpecCon-LoRA 消融实验详解
## 面向论文修改的证据与表述参考

数据快照：{STAMP}。新增正式消融完成于 2026-10-24，219/219、失败 0。模型为 RoBERTa-base；主要任务为 CoLA、MRPC、RTE。本文结合原组件实验、补充消融与独立机制验证，适合修改方法论证、实验分析、图表说明和局限性。

**最明确的结论：在默认 755,712 参数预算下，自适应分配相对同预算近均匀分配有平均优势；rank 置换进一步支持模块容量放置的重要性。**

**核心限制：一致性加权在原校准设置下有收益，但换五个校准子集后平均收益未复现；跨子集 rank 分配也没有更稳定。论文应把评分的结构预测性与下游稳定收益分开。**

{table(['论文主张','当前证据判断','主要依据'],[
['默认预算下分配有价值','可作为主要实验证据','等预算 SVD：+2.529 / +1.471 / +0.481'],
['容量放置位置重要','可作为主要辅助证据','MRPC +0.735；RTE +2.407'],
['一致性评分反映可重现结构','有较强描述性机制证据','控制混杂后的相关：0.980 / 0.974 / 0.963'],
['一致性加权提高性能','仅有条件支持','原设置均值为正；新校准子集均值为负'],
['SVD 初始化方向有价值','CoLA/MRPC 支持较好；RTE 较弱','同范数、同类行 Gram 随机正交对照'],
['更稳定、跨预算普遍更优','现有结果不支持','rank 距离增大；多预算结果有正有负']])}

阅读路径：第 02–03 节统一方法与协议；第 04–15 节逐项解读；第 16 节集中看图；第 17–19 节用于直接修改论文；第 20 节及附录用于追溯与核对。

本文未读取你的论文正文，因此给出的是**依据当前代码与实验的修改参考**，并非对原稿逐句审校。下文“支持较强”表示在现有证据中相对更直接，不代表已通过显著性检验。正向与负向结果均保留。
''')

page('02 方法组件与可检验主张','''
## 02 方法由哪些部分组成，消融分别验证什么

方法的输出是训练前确定的各模块 rank。块梯度的谱与子空间信息用于分配评分和初始化；训练阶段 LoRA 的 A/B 仍然更新，不能描述为始终固定在校准得到的方向。

1. 将校准数据分成 H 块，计算每块平均梯度，并提取其领先奇异值与左奇异子空间。
2. 用平均谱能量 μ 描述候选增量的能量，用跨块子空间投影重叠 C 描述方向的一致性。
3. 一致性调整后的能量为 μ × max(C, C_floor)^λ。
4. 候选增量评分为“调整后能量 × 相对能量^(γ−1) ÷ 参数成本”；单个 rank 的参数成本是输入维度加输出维度。
5. 在 rank 前缀可行性、上下界与参数预算约束下分配容量，并可使用梯度 SVD 初始化 A、将 B 置零。

| 组件 / 符号 | 对应问题 | 当前可以使用的对照 |
|---|---|---|
| 模块间自适应分配 | 相同参数量下怎样放置容量更好？ | 同预算近均匀对照；严格均匀多预算；同形状置换 |
| 一致性指数 λ | 相比只看分块谱能量，有没有额外收益？ | F−D、E−C、独立校准子集成对训练 |
| SVD 初始化 | 整套初始化、方向、尺度各有什么作用？ | F−E；同范数随机正交；Frobenius 匹配；方向固定改尺度 |
| 分块数 H、样本数 N | 校准粒度和校准规模如何影响结果？ | 固定其他设置的 H、N 扫描 |
| γ | 谱能量加权形式的指数如何影响结果？ | γ=0.7/1.0/1.2/1.5 |
| 参数成本除项 | 成本校正本身是否带来收益？ | 当前没有单独去掉成本除项的正式对照 |
| 训练正则 α | R-Drop 的 KL 项是否有贡献？ | 双前向 CE vs 双前向 CE+KL |

λ=0 仍保留同样的分块谱计算，只移除一致性加权；关闭整个一致性计算路径可能改成“平均梯度的谱”，会引入另一个变化。γ=1 只是去掉额外的相对能量指数因子，谱能量评分仍然存在。

λ、LoRA dropout 概率与 R-Drop α 是三个不同量，论文符号与表头应分别定义。公式与术语依据项目方法说明；实验复现应使用各批次保存的冻结源码。
''')

hp={t:next(x['hyperparameters'] for x in M['runs'] if x['task']==t and x['seed']==42) for t in TASKS}
page('03 实验协议与统计口径',f'''
## 03 实验协议、完成情况与统计口径

共同设置：RoBERTa-base、FP32、batch=32、冻结分类头、LoRA scaling=2、rank 范围 [2,16]，固定末轮验证集指标。普通配置使用种子 42、777、2026；训练数据顺序和训练前 CPU/CUDA 随机状态与原正式组件实验配对。

{table(['设置','CoLA','MRPC','RTE'],[[label]+[hp[t][key] for t in TASKS] for key,label in [('epochs','训练轮数'),('lr','学习率'),('lambda','完整方法 λ'),('wd','Weight decay'),('dropout','LoRA dropout'),('label_smoothing','Label smoothing'),('max_len','最大长度'),('warmup','Warmup 比例')]])}

默认 H=4、N=512（16×32）、γ=1.2、R-Drop α=1。CoLA 为 MCC×100，MRPC/RTE 为 Accuracy×100。MRPC 的 Accuracy/F1 平均值不得混入本表。

{table(['新增实验组','正式训练数'],[['等预算',18],['初始化方向与尺度',27],['等前向次数 R-Drop',18],['分块 H',27],['校准 N',27],['γ',27],['预算曲线',45],['独立校准子集下游训练',30],['合计',219]])}

原组件为 54 个结果格；原 R-Drop 为 36 格；rank 置换新增 6 次并复用 6 个原结果。独立机制有 33 次校准与 3 次谱随机种子复核。存在复用，不能把这些数字简单相加成独立训练次数。新增结果文件另含 45 条复用参考；219+45=264 是记录数。

表中 ± 为样本标准差（ddof=1）。配对差值先按相同训练种子相减，再计算均值与标准差，不能用两组 SD 相减。五个校准 A 子集固定训练 seed42、彼此可能重叠，不能写成五训练种子。除明确说明外，下文结果依次为 CoLA/MRPC/RTE；不报告未经检验的 p 值或显著性。
''',True)

component_defs=[['A','均匀 rank8','随机 A / 零 B','1,327,104'],['B','均匀 rank8','SVD A / 零 B','1,327,104'],['C','分块谱，λ=0','随机 A / 零 B','755,712'],['D','分块谱，λ=0','SVD A / 零 B','755,712'],['E','分块谱＋一致性','随机 A / 零 B','755,712'],['F','分块谱＋一致性','SVD A / 零 B','755,712']]
pair_names=[('F','D'),('E','C'),('B','A'),('D','C'),('F','E')]
page('04 六组组件消融',f'''
## 04 六组组件消融：整体效果与各组件关系

目的：在三种分配方式与两种初始化之间建立对照。A/B 与 C–F 预算不同；C–F 之间及同一分配内部的初始化比较预算相同。

{table(['ID','分配','初始化','适配器参数'],component_defs)}

{table(['方案','CoLA','MRPC','RTE'],[[n]+[fmt(cf(t,n)) for t in TASKS] for n in 'ABCDEF'])}

{table(['同预算差值','CoLA','MRPC','RTE'],[[a+'−'+b]+[delta([x-y for x,y in zip(cf(t,a),cf(t,b))]) for t in TASKS] for a,b in pair_names])}

**解读。** F 的三任务均值均最高，支持完整组合在这套设置下的效果。F−D 支持 SVD 初始化下的一致性收益；F−E 支持完整分配下的整套 SVD 初始化。E−C 与 D−C 不普遍为正，因此不应写成各组件都独立、稳定地提升性能。

F 比 uniform rank8 少 43.06% 的适配器参数，同时均值更高，可描述为这组设置下的性能—参数量优势；但 A/B 与 C–F 的比较不能证明等预算下的纯分配收益。该问题已由第 05 节的新实验补上。

建议图表：正文保留 A–F 定义与参数列，性能表报告 mean±SD；不要只挑 seed42 的最大提升。来源：components.csv。
''',True)

page('05 精确等预算分配',f'''
## 05 精确等预算：最直接的容量分配证据

目的：排除参数量差异，检验当前分配方案是否优于近均匀分配。两组均使用 755,712 参数；近均匀方案固定 attention rank4、FFN rank5。其余训练设置配对，随机初始化与 SVD 初始化分别比较。

{summary_table([('near_uniform_random','近均匀＋随机'),('adaptive_random','自适应＋随机（E）'),('near_uniform_svd','近均匀＋SVD'),('full','自适应＋SVD（F）')])}

### SVD 初始化下的逐种子差值

{seed_delta_table('allocation_same_budget_svd')}

### 随机初始化下的逐种子差值

{seed_delta_table('allocation_same_budget_random')}

**结论。** SVD 初始化下 9 个配对为 8 胜 1 平，三个任务均值均提升。这是论文中最适合作为默认预算分配优势的证据。随机初始化下三个任务均值也为正，CoLA/MRPC 各三胜，RTE 两胜一负；分配收益并非只在 SVD 条件出现。

**边界。** 近均匀方案是一个预先固定的结构性基线，不是所有可能近均匀配置中调优后的最优者。该对照评估整套分配方案，不能把全部收益只归于一致性因子；一致性需看 F−D。不同 rank 下按同一规则初始化，会改变实际初始化矩阵的维度，因此应写“相同初始化规则”。

可用于正文：在 755,712 个适配器参数预算下，SpecCon-LoRA 相对采用相同 SVD 初始化规则的近均匀分配，在 CoLA、MRPC 和 RTE 上分别提高 2.529、1.471 和 0.481 个指标点；对应九个种子配对中八个提升、一个持平。
''')

interaction={t:[(comp[t,'F',s]-comp[t,'D',s])-(comp[t,'E',s]-comp[t,'C',s]) for s in SEEDS] for t in TASKS}
page('06 一致性与初始化的条件关系',f'''
## 06 一致性加权：原设置有效，但收益有条件

目的：在分块数、样本、谱随机性、γ、预算与训练配方固定时，仅比较 λ=0 与任务配置的 λ>0。CoLA/MRPC 的完整 λ=1.5，RTE 为 0.5，不是跨任务统一 λ。

{table(['任务 / 对照','seed42','seed777','seed2026','均值 ± SD'],[[LABEL[t]+' '+a+'−'+b]+[f'{comp[t,a,s]-comp[t,b,s]:+.3f}' for s in SEEDS]+[delta([comp[t,a,s]-comp[t,b,s] for s in SEEDS])] for t in TASKS for a,b in [('F','D'),('E','C')]])}

F−D 在 MRPC 三种子全正，在 CoLA/RTE 各有一个负差。尤其 CoLA 的均值 +1.632 受到 seed42 的 +5.729 影响，而 seed777 是 −1.728，不能仅展示最有利的种子。

E−C 的 CoLA 三种子全负，MRPC 均值仅 +0.082，RTE 均值 +1.083。这说明一致性收益与初始化条件相关。不能声称一致性在随机初始化下也普遍改善效果。

### 可补充报告的同预算交互描述

使用 C/D/E/F 这四组，可以计算 (F−D)−(E−C)：即一致性在 SVD 初始化下的收益，减去其在随机初始化下的收益。

{table(['任务','交互差值均值 ± SD','胜/平/负'],[[LABEL[t],delta(interaction[t]),wins(interaction[t])] for t in TASKS])}

该计算是本报告新增的**事后描述性整理**，不是新增训练，也不是预注册的显著性交互检验。三个均值为正可以说明条件差异，但不足以写成“证明存在普遍协同机制”。同预算 C–F 的比较优于跨预算的 A/B 与 C–F 交互解释。

论文应同时引用第 15 节的独立校准子集训练：那一实验中三个平均差值均为负。推荐表述为“原校准配置下观察到与初始化相关的一致性收益”，并明确对校准样本选择的依赖。
''')

page('07 初始化方向与尺度',f'''
## 07 初始化：把整套方案、方向和尺度分开

控制变量：固定完整方法的 rank 表、预算、训练随机状态与 B=0。新增 27 次训练分别替换 A 的初始化，不改变分配。

{table(['配置','具体变化'],[['随机（E）','标准随机 A、零 B'],['SVD（F）','梯度 SVD A、零 B'],['随机正交同范数','随机正交行空间；匹配 SVD Frobenius 范数和行 Gram 结构'],['随机同范数','Kaiming 随机 A 缩放到 SVD 范数；不匹配正交结构'],['SVD 同随机范数','保留 SVD 方向，只将 Frobenius 范数改为随机初始化范数']])}

{summary_table([('adaptive_random','随机（E）'),('full','SVD（F）'),('orthogonal_svdnorm','随机正交同范数'),('random_svdnorm','随机同范数'),('svd_randomnorm','SVD 同随机范数')])}

{contrast_table([('direction_same_gram','F−随机正交同范数'),('init_same_frobenius','F−随机同范数'),('svd_scale_only','F−SVD同随机范数'),('random_scale_only','随机同范数−E')])}

**方向。** 最直接的是同范数随机正交控制。CoLA 三种子全胜；MRPC 两胜一负；RTE 一胜两负，RTE 正均值主要来自 seed42，不能写成稳定收益。仅匹配 Frobenius 的随机控制在 RTE 均值反而略高于 F（F−控制 = −0.120）。

**尺度。** 保留 SVD 方向时，改回随机范数几乎不改变 CoLA 均值，但 MRPC/RTE 的均值下降；随机方向改尺度则是 CoLA/RTE 上升、MRPC 下降。尺度与方向的作用有任务依赖性。

正文建议：用“控制范数与行正交结构后的方向比较支持 CoLA/MRPC 上的梯度方向价值”，而不是把原 F−E 的全部收益归结为方向。初始化应作为分配方法的辅助组件，不能替代分配器本身的证据。
''',True)

shuffle={}
for t in ['mrpc','rte']:
 for method in ['original','permuted']:
    shuffle[t,method]=[float(next(x['score'] for x in SH if x['task']==t and x['placement']==method and int(x['seed'])==s)) for s in SEEDS]
page('08 rank 放置位置',f'''
## 08 同形状 rank 置换：容量放在哪里是否重要

目的：检验完整方法的优势是否仅来自非均匀 rank 的分布，还是来自这些 rank 被放到具体模块的位置。置换严格限制在输入、输出形状相同的模块组内，保持每组 rank 多重集合、总参数量 755,712 和训练配方。

{table(['任务','原始分配','置换分配','原始−置换'],[[LABEL[t],fmt(shuffle[t,'original']),fmt(shuffle[t,'permuted']),delta([a-b for a,b in zip(shuffle[t,'original'],shuffle[t,'permuted'])])] for t in ['mrpc','rte']])}

{table(['任务','训练种子','置换种子','原始分数','置换分数','差值'],[[LABEL[t],s,ps,f'{shuffle[t,"original"][i]:.3f}',f'{shuffle[t,"permuted"][i]:.3f}',f'{shuffle[t,"original"][i]-shuffle[t,"permuted"][i]:+.3f}'] for t in ['mrpc','rte'] for i,(s,ps) in enumerate(zip(SEEDS,[101,202,303]))])}

**结果。** MRPC 两个种子提升、一个下降，均值 +0.735；RTE 三种子均提升，均值 +2.407。RTE 对容量位置的证据更一致。

**实现与归因。** 置换后，各模块从自己的梯度按同一规则重新生成 SVD 初始化，没有把另一个模块的 A 矩阵直接搬过来。因此验证的是“rank 放置及随之产生的初始化/训练结果”；不能把这个对照描述为保留所有初始权重完全相同。

**统计边界。** 每个训练种子对应一个置换种子，训练随机性与置换随机性没有完全交叉，不能分别估计两者方差。只有两项任务，也没有完成 CoLA 下游置换。机制实验中的 100 次置换是投影评价，不是 100 次下游训练。

正文可用：在保持 rank 集合与参数预算不变的同形状置换对照中，原始分配在 MRPC 和 RTE 上分别领先 0.735 和 2.407 个指标点，其中 RTE 的三个种子均为正差值。该结果支持模块容量放置的作用。
''')

rdlookup={(x['task'],x['method'],int(x['alpha']),int(x['seed'])):float(x['score']) for x in RD}
rdrows=[]
for method,label,pref in [('uniform','Uniform r8','uniform8'),('speccon','SpecCon','speccon')]:
 for condition in ['single','two','kl']:
    name={'single':'单前向 CE，α=0','two':'双前向 CE，α=0','kl':'双前向 CE+KL，α=1'}[condition]
    v=[]
    for t in TASKS:
        if condition=='two':a=vf(t,pref+'_two_ce')
        else:a=[rdlookup[t,method,0 if condition=='single' else 1,s] for s in SEEDS]
        v.append(fmt(a))
    rdrows.append([label,name]+v)
page('09 R-Drop 正则化与计算次数',f'''
## 09 R-Drop：整套配方与 KL 项的贡献

原 α=0/1 实现同时改变前向次数：α=0 一次前向，α=1 两次前向。因此原始对照只能解释整套 R-Drop 配方。新增双前向 CE、α=0 对照，使前向次数一致，从而更直接检验 KL 项。

{table(['方法','训练方式','CoLA','MRPC','RTE'],rdrows)}

{contrast_table([('speccon_KL_same_compute','SpecCon：CE+KL−双CE'),('uniform_KL_same_compute','Uniform：CE+KL−双CE')])}

### SpecCon 的 KL 项：逐种子差值

{seed_delta_table('speccon_KL_same_compute')}

**结论。** 保留两次前向时，KL 项在 SpecCon 三任务的均值增益为 +2.042、+1.961、+1.203。MRPC 两胜一平，CoLA/RTE 各有一个负差。支持正则化配方对最终分数有贡献，但不能声称每个种子均提升。

**用词。** “等计算”在这里主要指相同训练步数与相同前向次数；KL 本身仍有运算，因此不能写成实测运行时间、FLOPs 完全相等。Uniform 与 SpecCon 的预算不同，应分别在方法内部归因 KL 项。

**与方法主张的关系。** 原组件 A–F 均使用相同 α=1，因此 F−D、F−E 等对照不是由 α 开关造成。该实验说明 R-Drop 本身贡献了收益，不能把全部最终分数优势都归给分配器。未来若要证明一致性收益与正则完全独立，还需在关闭 KL 时成对比较 F 与 D；现有表格未覆盖这一交叉问题。
''',True)

lamrows=[]
for t in ['cola','mrpc','rte','qnli']:
 for lam in sorted(set(float(x['lambda_value']) for x in LAM if x['task']==t)):
  v=[float(x['score']) for x in LAM if x['task']==t and float(x['lambda_value'])==lam]
  lamrows.append([LABEL.get(t,t.upper()),f'{lam:g}',len(v),fmt(v)])
page('10 λ 敏感性与未完成的搜索',f'''
## 10 λ 敏感性：任务依赖与正在搜索的边界

λ 控制一致性加权强度。λ=0 仍保留分块谱，其余配置与对应任务匹配。以下采用已有完整三种子结果，不用单种子最佳点替换。

{table(['任务','λ','n','均值 ± SD'],lamrows)}

**RTE。** λ=0/0.5/1.5 的均值为 78.700/80.505/79.422，支持在已测点中选择 0.5，而非一致性权重越大越好。

**QNLI。** λ=0 的均值 93.032 高于 0.7 的 92.910 和 1.5 的 92.849。主表按既有约定使用非零 λ=0.7，但不能据此声称一致性在 QNLI 上带来平均提升。

八任务 seed42 配对中，MNLI、QQP 和 QNLI 的 λ=1.5 略降，STS-B 基本不变；SST-2 小幅提高。这些主要是单种子边界观察，不能与三种子均值混作同等证据。

### 最新 QNLI/QQP 搜索：仅记录快照

读取队列状态时间：{HPO['updated']}。新增筛选完成 {len(HPO['completed'])}/{HPO['new_screening_total']}，另有复用参考；新增候选的多种子确认尚未开始。搜索包含 λ 单因素和学习率×轮数两个组，不是三者完整笛卡尔积。

当时 seed42 暂列第一的非零候选均为 λ=0.25：QNLI 93.2638、QQP 91.5508。它们是验证集筛选结果，不能直接与旧配置三种子均值比较，也不能当作独立测试效果。最终应等预定确认种子完成后再更新论文。快照源文件已保存在 sources/hpo_report.md。
''',True)

page('11 分块数与校准规模',f'''
## 11 H 与 N：当前配置合理，不代表不敏感

### 分块数 H：固定总校准样本 N=512

{summary_table([('H2','H=2'),('full','H=4（默认）'),('H8','H=8'),('H16','H=16')])}

H=4 在三个任务的已测均值中最高。H=8 相对默认在 CoLA/MRPC 分别下降 2.535/1.389，在 RTE 仅下降 0.120；H=16 的三个均值均下降。说明更多分块并不必然带来收益。

固定 N 改 H 会同时改变每块样本数、梯度累加的数值路径和谱近似随机数的分配。适合解释为整体分块敏感性，而不是完全隔离了某一个理论因素。

### 校准规模 N：固定 H=4

{summary_table([('N128','N=128'),('N256','N=256'),('full','N=512（默认）'),('N1024','N=1024')])}

N=512 在三任务的已测均值中最高，N=1024 没有继续改善。以嵌套训练集前缀增加 N 会改变样本构成与每块样本数；结果不支持“更多校准样本保证更高性能”。

**论文怎么写。** 在冻结训练配方与当前候选网格内，H=4、N=512 取得较好的跨任务均值表现。不同设置的差值与标准差提示对校准配置存在敏感性，应报告完整曲线而非仅标出最优点。

**不能怎么写。** 不应称 H/N 几乎无影响，不应因默认配置最好就宣称已找到普遍最优；网格有限，默认配置来自此前保留的训练设置。N=512 的样本效率也不能直接推出总训练时间更低，需另报前处理和训练的实测成本。
''')

budgetrows=[]
for rank in [3,6,8]:
 for which in ['uniform','adaptive']:
  name='uniform8_svd' if rank==8 and which=='uniform' else f'budget_r{rank}_{which}'
  budgetrows.append([f'{165888*rank:,}',('均匀' if which=='uniform' else '自适应')]+[vals(t,name) for t in TASKS])
page('12 γ 与参数预算',f'''
## 12 γ 与预算曲线：避免把局部优势扩大为普遍优势

### γ 扫描

{summary_table([('gamma0.7','γ=0.7'),('gamma1.0','γ=1.0'),('full','γ=1.2（默认）'),('gamma1.5','γ=1.5')])}

γ=1.2 在 MRPC/RTE 的已测点中最好；CoLA 的 γ=1.0 比默认高 0.257。默认相对 γ=1.0 的差值为 −0.257/+1.225/+0.963，支持折中选择，不能声称每项任务都最优。γ=1 仍然使用边际谱能量与参数成本校正。

### 严格均匀 rank3/6/8 对照

{table(['参数预算','分配','CoLA','MRPC','RTE'],budgetrows)}

{contrast_table([('budget_rank3_allocation','rank3预算：自适应−均匀'),('budget_rank6_allocation','rank6预算：自适应−均匀'),('budget_rank8_allocation','rank8预算：自适应−均匀')])}

rank3 预算下 CoLA/MRPC 均值为正、RTE 持平；rank6/8 下 CoLA/MRPC 均值为负、RTE 为正。只有当前默认 755,712 预算的近均匀对照在三任务均值上全部为正。

预算曲线固定学习率、训练轮数和 rank 范围 [2,16]，没有逐预算调参。这回答同一配方下的表现，不代表每个方法充分调参后的最优边界。参数少 43.06% 只对应默认 SpecCon 与 uniform rank8 的适配器参数比较，不能延伸为显存或端到端耗时也减少 43.06%。
''',True)

page('13 独立机制：评分预测性','''
## 13 独立校准机制：评分是否包含可复现结构

设计：每任务固定一个 B 子集，五个 A 子集各有两种样本顺序。A/B 在样本 ID 和完全相同文本层面隔离，均来自训练集；A 之间允许重叠。每次 512 条、H=4、预算 755,712，固定分类头 seed42。主分析使用预先指定的 order0。

先在 A 上计算一致性评分，再评价 A/B 的梯度子空间重现程度。对候选 rank k=3…16，在相同 k、矩阵形状组内进行秩转换，并控制 log 累计块谱能量；进一步加入模块类型控制。

| 任务 | 控制 k/形状/能量 | 再控制模块类型 | 与独立 B 自身一致性的相关 |
|---|---|---|---|
| CoLA | 0.980 ± 0.004 | 0.953 ± 0.020 | 0.956 ± 0.010 |
| MRPC | 0.974 ± 0.007 | 0.949 ± 0.019 | 0.919 ± 0.013 |
| RTE | 0.963 ± 0.011 | 0.932 ± 0.027 | 0.917 ± 0.021 |

**可支撑的命题。** 一致性评分能够表征独立校准数据上可重现的梯度子空间结构。控制能量、维度与 rank 后仍然较强的关联，为其作为结构信息提供依据，而不仅是校准集内部的自洽指标。

### 谱随机种子复核

固定 B 样本与其他设置，将 B 的随机 SVD 种子从 3407 改为 103407，额外执行三个 B 校准。

| 任务 | 原相关 | B 独立谱种子相关 |
|---|---|---|
| CoLA | 0.9805 | 0.9804 ± 0.0043 |
| MRPC | 0.9737 | 0.9745 ± 0.0066 |
| RTE | 0.9628 | 0.9641 ± 0.0109 |

结果保持，减少了“共享谱随机种子导致关联”的疑虑。但这不是重新抽取独立 B 数据；每任务仍只有一个 B 子集、一个分类头初始化，五个 A 重复之间也存在关联。

**写作边界。** 不将相关系数称为准确率提升，不推断因果效应，不推断最终分配必然更稳定。正文可写“提供可复现结构信息”，避免直接写“证明评分能够稳定识别最优下游 rank”。
''')

page('14 独立机制：稳定性与捕获','''
## 14 rank 稳定性与独立梯度捕获

### rank 重复性：当前结果不支持普遍改善

距离定义为 Σ 参数成本×|两次 rank 差| / (2P)，越小越稳定。跨子集是五个 A 形成的十个相关配对的均值；换序是五组配对。

| 任务 | 跨子集 λ=0 | 跨子集完整 | 换序 λ=0 | 换序完整 |
|---|---:|---:|---:|---:|
| CoLA | 0.0962 | 0.1118 | 0.0738 | 0.0667 |
| MRPC | 0.0598 | 0.0650 | 0.0970 | 0.0992 |
| RTE | 0.1710 | 0.1970 | 0.1041 | 0.1201 |

跨子集距离三任务均增大；相同样本换序时只有 CoLA 改善。由于校准采用 train 模式，换序也改变样本对应的 dropout 掩码，不能解释为纯粹的分块随机性。

### 独立 B 梯度捕获：局部代理收益有限

每个 A 使用共同的 SVD 基底，只改变 rank 表。左捕获为 Σ||UᵀG_B||² / Σ||G_B||²，右捕获为 Σ||G_BV||² / Σ||G_B||²。右侧对应实际 SVD-A 初始化的行空间。

| 任务 | 左捕获：λ=0→完整 (%) | 左增量 (pp) | 右捕获：λ=0→完整 (%) | 右增量 (pp) |
|---|---|---:|---|---:|
| CoLA | 96.494→96.459 | −0.0355 | 99.753→99.743 | −0.0098 |
| MRPC | 96.141→96.146 | +0.0055 | 99.729→99.733 | +0.0034 |
| RTE | 81.973→82.235 | +0.2617 | 99.186→99.187 | +0.0008 |

RTE 左捕获的提升最清楚；右捕获已超过 99%，新增差值很小，且 CoLA 两个方向都略降。换 order1 后，MRPC 左捕获增量变为负，说明细微收益也依赖样本顺序。

完整方法相对同预算近均匀和 rank 置换的捕获均值较高，可作为整套分配的局部补充证据，但无法把全部差异单独归给一致性项。

**必须收窄的主张。** 不写“一致性普遍稳定秩分配”“一致性普遍提升梯度捕获”“捕获率等于能力保留率”。离散预算竞争可能放大评分扰动只是待验证解释，不能当作本轮已证实机制。
''')

subset_summary=[]
subset_data={}
for t in TASKS:
 for arm in [0,1]:
  v=[float(next(x['score'] for x in R['runs'] if x['task']==t and x['group']=='calibration_subset' and int(x['replicate'])==i and (x['consistency_lambda']>0)==bool(arm))) for i in range(5)]
  subset_data[t,arm]=v
 subset_summary.append([LABEL[t],fmt(subset_data[t,0]),fmt(subset_data[t,1]),delta(contrast(t,'independent_subset_full_minus_lambda0')),wins(contrast(t,'independent_subset_full_minus_lambda0'))])
page('15 独立校准子集的下游训练',f'''
## 15 最关键的边界：换校准样本后，增益没有复现

设计：复用独立机制实验保存的五个 A 子集梯度与块谱，λ=0 与完整方法成对训练。三任务×五子集×两设置 = 30 次正式训练，模型与训练种子固定 seed42。B 仅用于既有机制评价，不参与分配。下游仍使用完整训练集，不能称作未见数据测试。

{table(['任务','λ=0，均值±SD','完整，均值±SD','配对差值±SD','胜/平/负'],subset_summary)}

{table(['校准子集','CoLA 差值','MRPC 差值','RTE 差值'],[[f'A{i}']+[f'{subset_data[t,1][i]-subset_data[t,0][i]:+.3f}' for t in TASKS] for i in range(5)])}

**结论。** 三任务的平均差值为 −0.298、−0.784、−0.361。MRPC 四个子集下降、一个持平；CoLA 两胜三负；RTE 一胜四负。这限制了原校准配置中 F−D 正收益的可迁移范围，应作为正文讨论的关键限制。

**不是五训练种子。** 五个校准子集可能重叠，且共享模型/训练 seed42。不能据此推出一致性在所有设置无效，也不宜把该组 SD 与主实验三训练种子的 SD 当成同一种稳定性度量。

**性能波动与平均收益分开。** CoLA/RTE 的跨子集分数 SD 略小，但 MRPC 变大，且三个均值均下降。因此不能用局部方差下降来证明整体更可靠。

**与机制实验的关系。** 高评分与独立结构重现相关，不意味着按该评分调整 rank 后一定提高验证集表现。离散分配、初始化、训练优化与任务目标之间仍存在未被局部代理指标覆盖的环节；这里只能提出解释方向，不能宣布因果链已建立。

可用于局限性：一致性加权在原校准配置下带来收益，但在固定训练种子、替换五个校准子集的成对实验中未保持正的平均增益。这表明其下游效果依赖校准样本，结构预测性尚未普遍转化为性能与分配稳定性的改善。
''')

# Figures are standalone, exportable scientific artifacts; no significance marks.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
colors=['#176a92','#ce6b31','#55834d']
fig,axes=plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
for ax,t in zip(axes,TASKS):
 for method,label,color in [('uniform','Uniform SVD',colors[0]),('adaptive','Adaptive SVD',colors[1])]:
  v=[vf(t,'uniform8_svd' if r==8 and method=='uniform' else f'budget_r{r}_{method}') for r in [3,6,8]]
  ax.errorbar(np.array([3,6,8])*165888/1e6,[st.mean(x) for x in v],yerr=[st.stdev(x) for x in v],marker='o',capsize=3,lw=1.3,color=color,label=label)
 ax.scatter([.755712],[st.mean(vf(t,'full'))],marker='*',s=110,color=colors[1],zorder=5,label='Default adaptive')
 ax.scatter([.755712],[st.mean(vf(t,'near_uniform_svd'))],marker='D',s=35,facecolors='none',edgecolors=colors[0],zorder=5,label='Near-uniform')
 ax.set_title(LABEL[t]);ax.set_xlabel('Adapter parameters (M)');ax.set_ylabel('MCC x100' if t=='cola' else 'Accuracy x100');ax.grid(alpha=.2)
axes[0].legend(fontsize=7,loc='best')
fig.savefig(ASSETS/'budget_curve.png',dpi=190);fig.savefig(OUT/'预算曲线_可投稿图.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(11,2.9),layout='constrained')
for ax,names,xs,title in zip(axes,[['H2','full','H8','H16'],['N128','N256','full','N1024'],['gamma0.7','gamma1.0','full','gamma1.5']],[[2,4,8,16],[128,256,512,1024],[.7,1.,1.2,1.5]],['Block count H','Calibration size N','Gamma']):
 for t,col in zip(TASKS,colors):ax.plot(range(4),[st.mean(vf(t,n))-st.mean(vf(t,'full')) for n in names],marker='o',label=LABEL[t],color=col,lw=1.3)
 ax.axhline(0,color='#6a7380',lw=.8,ls='--');ax.set_xticks(range(4),list(map(str,xs)));ax.set_xlabel(title);ax.set_ylabel('Mean delta vs default');ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.savefig(ASSETS/'sensitivity.png',dpi=190);fig.savefig(OUT/'超参数敏感性_可投稿图.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(11,2.7),layout='constrained')
for ax,t,col in zip(axes,TASKS,colors):
 d=contrast(t,'independent_subset_full_minus_lambda0');ax.bar(range(5),d,color=col,alpha=.8);ax.axhline(0,color='#6a7380',lw=.8);ax.set_xticks(range(5),[f'A{i}' for i in range(5)]);ax.set_title(LABEL[t]);ax.set_ylim(-2,1.6);ax.set_ylabel('Full minus lambda=0');ax.grid(axis='y',alpha=.2)
fig.savefig(ASSETS/'calibration_deltas.png',dpi=190);fig.savefig(OUT/'独立校准子集差值_可投稿图.pdf');plt.close(fig)

page('16 图形总览与图注','''
## 16 预算与敏感性图：可直接作为制图起点

![预算曲线](assets/budget_curve.png)

**图 1 建议图注。** 不同参数预算下的自适应 SVD 与均匀 SVD。实线仅连接严格均匀 rank3/6/8 的匹配预算；星形和空心菱形分别表示默认预算下的自适应与近均匀分配，不计入严格均匀曲线。误差条为三训练种子的样本标准差，不是置信区间。各预算使用冻结训练配方。

![超参数敏感性](assets/sensitivity.png)

**图 2 建议图注。** H、N、γ 的单因素敏感性，纵轴为该任务三种子均值减默认配置均值。默认 H=4、N=512、γ=1.2。此图只显示均值；各配置 SD 见第 11–12 节。零线不是显著性阈值。

正文图只需选最能支撑主线的部分，其余置于附录。不要只绘制获益预算或删除负向的配置点。两幅图均另存为矢量 PDF，便于后续修改标题、字号和论文版式。
''',True)

page('17 论文主张与组织方式','''
## 17 论文应该怎样组织这些证据

### 主张与措辞的逐项修订

| 容易写过头的表述 | 建议改成 | 主要证据 |
|---|---|---|
| 一致性让 rank 分配更稳定 | 评分反映可复现子空间；分配重复性未普遍改善 | 第 13–15 节 |
| 所有任务、预算都优于均匀分配 | 默认预算下三任务均值更高，其他预算有任务依赖性 | 第 05、12 节 |
| SVD 初始化收益完全来自方向 | 方向与尺度分别有贡献，方向控制在 CoLA/MRPC 较有支持 | 第 07 节 |
| 每个组件都独立有效且协同 | 完整组合表现较好；一致性收益取决于初始化与校准样本 | 第 04、06、15 节 |
| 99% 捕获意味着保留 99% 能力 | 右侧梯度捕获为初始化处的局部投影比例 | 第 14 节 |
| 参数少就证明训练更快 | 默认适配器参数少 43.06%；运行成本需另外测量 | 第 12、19 节 |
| 使用五个随机种子验证校准鲁棒性 | 五个可能重叠的校准子集，训练种子固定为 42 | 第 15 节 |

### 正文推荐顺序

1. 定义方法与核心研究问题：固定预算下，利用谱能量与跨块几何信息进行一次性模块容量分配。
2. 给出完整组件表与精确等预算表，先建立效果证据。
3. 用 rank 置换说明容量位置的作用，再用方向控制解释辅助初始化。
4. 用独立评分预测性说明结构信息，再立即交代稳定性与独立子集训练的限制。
5. H/N/γ、完整预算曲线、R-Drop、完整种子与来源放附录；正文概括其边界。

### 适合强调的贡献

方法贡献可以围绕“跨块子空间一致性与考虑参数成本的谱增量评分的组合设计”。经验贡献可以围绕“默认预算下的分配效果与容量放置证据”。机制贡献应限于“独立样本上的结构预测性”，避免承诺尚未观察到的稳定性优势。

这是一种与当前证据一致的论证定位，不构成对相关工作新颖性的完整审查。若论文要声称优于其他方法，应另外审查相同预算、配方、调参机会与评测协议。
''')

page('18 可用于正文的中英文段落','''
## 18 可改写进正文的段落

以下段落针对本报告已验证的数据，可按论文上下文调整。它们不包含显著性、跨模型或未完成结果的承诺。

### 中文：分配与组件消融

在 755,712 个适配器参数的固定预算下，我们将 SpecCon-LoRA 与结构性近均匀分配进行比较。两组采用相同的 SVD 初始化规则与训练配方，SpecCon-LoRA 在 CoLA、MRPC 和 RTE 上的三种子均值分别提高 2.529、1.471 和 0.481 个指标点，九个配对中八个提升、一个持平。同形状 rank 置换进一步使 MRPC 和 RTE 的均值分别降低 0.735 和 2.407 个指标点，支持模块容量放置对性能的作用。

在原校准设置下，加入一致性加权相对 λ=0 在三个任务上分别提高 1.632、1.716 和 1.805 个指标点。然而，随机初始化下对应差值为 −0.799、0.082 和 1.083，表明收益依赖初始化条件。初始化方向控制进一步显示，在匹配范数与行 Gram 结构后，SVD 方向在 CoLA/MRPC 上具有更一致的正向证据，而 RTE 的结果波动较大。

### English: allocation and placement

Under a fixed adapter budget of 755,712 parameters, SpecCon-LoRA improves the three-seed mean over a structurally near-uniform allocation by 2.529, 1.471, and 0.481 metric points on CoLA, MRPC, and RTE, respectively, using the same SVD initialization rule and training recipe. Eight of the nine paired comparisons improve and one ties. Permuting ranks within equal-shape module groups, while preserving the rank multiset and parameter budget, reduces mean performance by 0.735 points on MRPC and 2.407 points on RTE. These results support the value of module-specific capacity placement under the evaluated setting.

### English: mechanism and limitations

After controlling for candidate rank, matrix shape, and spectral energy, calibration consistency remains strongly associated with subspace reproducibility on a disjoint calibration subset. This predictive relationship does not translate into universally more stable allocations or downstream gains. In paired downstream runs over five calibration subsets with training seed 42 fixed, consistency weighting yields mean differences of −0.298, −0.784, and −0.361 points on CoLA, MRPC, and RTE. These subsets may overlap, so the results are descriptive and do not constitute five independent training-seed replications.

术语建议：CoLA 使用 “MCC points”，MRPC/RTE 使用 “accuracy points”；合并叙述时用 “metric points”。避免把三个任务都叫 accuracy，也避免把均值差写成相对百分比提升。
''',True)

page('19 后续优先级与审稿问题','''
## 19 后续修改与补证优先级

以下是基于现有证据的建议，不表示这些实验已经完成，也不在本次任务中启动训练。

| 优先级 | 建议补证 | 回答什么问题 |
|---|---|---|
| 高 | 多训练种子×预先固定校准子集的 λ=0/完整成对训练 | 第 15 节的负均值是否依赖单个训练种子；校准与训练随机性各占多少 |
| 高 | 独立重抽 B、改变分类头初始化的评分预测性检查 | 当前高相关是否仅依赖一个 B 与一个模型初始化 |
| 中 | 能量稳定性替代子空间一致性，给予相同调参机会 | 几何方向一致性是否比能量幅值稳定性提供额外价值 |
| 中 | 保持预算约束，仅去掉评分中的参数成本除项 | 成本校正本身是否必要 |
| 中 | 相同设备和并发条件下测校准、初始化、训练、端到端时间与峰值显存 | 一次性分配的实际成本与效率 |
| 视论文主张 | 第二个模型上的关键同预算与 λ 对照 | 是否能够支持跨模型适用性 |

### 常见审稿问题的回答边界

**“收益是不是参数更多造成的？”** 默认等预算实验中两组均为 755,712 参数，可以直接回答；旧 A/B 与 C–F 跨预算表本身不能单独回答。

**“是不是只有 SVD 初始化在起作用？”** 随机初始化下的等预算比较均值仍为正；但是一致性本身的增益确实依赖初始化，需同时报告 E−C。

**“为什么评分相关很高，但新校准子集性能下降？”** 两者评价目标不同：一个是初始化处的结构重现，一个是训练后验证性能。可以说明未建立普遍的因果转化，但不能虚构已证实的噪声抑制或优化机制。

**“是否挑选最有利配置？”** 当前主配置来自此前选择，补充实验冻结了协议。报告所有已测网格、固定末轮与完整种子；正在进行的 HPO 需另标为验证集搜索，保留候选与确认步骤。

**“是否更快？”** 不以当前杂合并发日志作公平速度结论，不把 R-Drop 的双前向成本省略。219 次结果中的 elapsed_seconds 可以做运行记录，不能直接当统一硬件负载下的算法速度比较。

论文提交前优先修正证据与措辞不匹配的部分，再根据主张决定是否补实验。即使不新增训练，也应完整披露已观察到的负向结果。
''')

source_short=[
('S1','新增原始结果','experiments/extended_ablation_20261021_v2/results.json'),
('S2','新增配对统计','experiments/extended_ablation_20261021_v2/analysis/paired_results.json'),
('S3','新增协议','experiments/extended_ablation_20261021_v2/PROTOCOL.md'),
('S4','组件逐种子','reports/all_ablation_summary_20261020/01_组件与初始化/逐种子结果.csv'),
('S5','R-Drop 逐种子','reports/all_ablation_summary_20261020/03_RDrop消融/RDrop_36格逐种子.csv'),
('S6','λ 逐种子','reports/all_ablation_summary_20261020/02_lambda敏感性/lambda_全部可核验逐种子.csv'),
('S7','rank 置换','reports/all_ablation_summary_20261020/04_rank置换/rank置换_逐种子.csv'),
('S8','独立机制','reports/independent_calibration_mechanism_20261019/RESULTS_zh.md'),
('S9','未完成搜索','experiments/qnli_qqp_hparam_sweep_20261022_v1/analysis/REPORT.md')]
page('20 来源与更新方式',f'''
## 20 来源、审计与后续更新

根目录：/workspace/SpecCon-LoRA/。下表路径均相对于根目录。本报告同时保留 sources 快照与逐文件 SHA-256；后续源队列变化不会悄悄改变本版 PDF。

{table(['编号','内容','原路径'],source_short)}

### 本次检查

- 新增结果共 219 条正式运行、45 条复用参考；原组件 54 条记录。
- 核对新增配对文件所记录的原结果 SHA-256，重新计算所有配置均值、样本 SD 与普通三种子配对差值。
- 本报告新增的同预算交互差值由原 C/D/E/F 结果计算，明确标为事后描述性分析。
- PDF 检查中文提取、严格解析、逐页独立渲染；图形另存矢量 PDF。

### 编辑与更新

同目录提供 Markdown 与 Word，可复制分析文字；Excel 包含新增逐次记录、配置均值、配对差值、旧组件、R-Drop、λ、置换和来源索引。附录提供三种子数据，原始 metrics 路径保留在 Excel/sources 中。

后续 QNLI/QQP 搜索完成时，先区分单种子筛选与预定多种子确认，再更新第 10 节。若增加校准重复或新模型，应另建实验组，不能将不同重复单位直接合并为一个更大的 n。

保留原四页简版；本目录为独立详细版。本文未改变任何训练配置，也未启动、停止或恢复训练进程。
''',True)

# Complete result appendix: group rows across seeds, with calibration replicates kept explicit.
for t in TASKS:
 names=sorted({r['name'] for r in allrows if r['task']==t},key=lambda n:(n.startswith('subset'),n))
 appendix=[]
 for name in names:
  subsetrows=[r for r in allrows if r['task']==t and r['name']==name]
  values={int(r['seed']):float(r['score']) for r in subsetrows}
  display=name.replace('near_uniform','nearUniform').replace('budget_','').replace('orthogonal_','orth_').replace('adaptive_random','adaptiveRnd').replace('uniform8_random','uniform8Rnd').replace('uniform8_svd','uniform8SVD')
  appendix.append([display]+[f'{values[s]:.3f}' if s in values else '—' for s in SEEDS]+[fmt(list(values.values())),f'{subsetrows[0]["budget"]:,}'])
 page('附录 '+LABEL[t]+' 全部新增与参考配置',f'''
## 附录：{LABEL[t]} 全部新增与复用参考

来源为新增批次的 runs+references。subset 行仅 seed42，各行是一对子集实验的一臂；不能把同任务十个 subset 行当作十训练种子。其余普通配置均 n=3。指标为 {'MCC×100' if t=='cola' else 'Accuracy×100'}。

{table(['配置（缩写）','42','777','2026','均值 ± SD','参数'],appendix)}

orth_=随机正交；Rnd=随机；nearUniform=attention4/FFN5；r3/r6/r8=严格等价预算；full=完整方法 F。完整配置名与来源路径见 Excel。原六组组件的 C 行等不在新增参考集合内，另见下一附录。
''',True)

page('附录 原六组组件逐种子',f'''
## 附录：原六组组件的完整训练种子

CoLA 为 MCC×100，MRPC/RTE 为 Accuracy×100。A/B 的参数量为 1,327,104；C–F 为 755,712。每行 n=3，固定末轮。

{table(['任务','方案','seed42','seed777','seed2026','均值 ± SD'],[[LABEL[t],v]+[f'{comp[t,v,s]:.3f}' for s in SEEDS]+[fmt(cf(t,v))] for t in TASKS for v in 'ABCDEF'])}

### 独立校准子集训练的单次成绩

{table(['任务','设置','A0','A1','A2','A3','A4'],[[LABEL[t],('λ=0' if arm==0 else '完整')]+[f'{x:.3f}' for x in subset_data[t,arm]] for t in TASKS for arm in [0,1]])}

五个 A 子集共享训练 seed42，并非独立测试集；此处列出完整单次成绩便于复核第 15 节。配对差值与来源路径见 Excel 和 JSON 快照。
''',True)

assert len(pages)==24

# Export all tables to an editable workbook.
wb=Workbook();wb.remove(wb.active)
def sheet(name,data):
 ws=wb.create_sheet(name)
 fields=[]
 for r in data:
  for k in r:
   if k not in fields:fields.append(k)
 ws.append(fields)
 for r in data:
  values=[]
  for k in fields:
   value=r.get(k)
   if isinstance(value,(dict,list)):value=json.dumps(value,ensure_ascii=False)
   elif isinstance(value,str) and k in ['seed','score','parameters','alpha','epochs','lambda_value','budget','lr','lam','mean','sample_std','n','replicate'] and re.fullmatch(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?',value):
    value=float(value) if any(c in value for c in '.eE') else int(value)
   values.append(value)
  ws.append(values)
 ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 for c in ws[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='234B65')
 for col in ws.columns:
  width=min(60,max(12,max(len(str(c.value or '')) for c in list(col)[:100])+2));ws.column_dimensions[col[0].column_letter].width=width
 for row in ws.iter_rows(min_row=2):
  for c in row:
   if isinstance(c.value,float):c.number_format='0.000000'
   c.alignment=Alignment(vertical='top')
sheet('新增219次',R['runs']);sheet('复用45条',R['references']);sheet('配置均值',P['summaries']);sheet('配对差值',P['contrasts']);sheet('组件54格',COMP);sheet('RDrop36格',RD);sheet('lambda全部',LAM);sheet('rank置换',SH);sheet('HPO快照',HPO_CANDIDATES);sheet('来源',source_records)
sheet('同预算交互_事后计算',[dict(task=t,seed=s,interaction_difference=interaction[t][i]) for t in TASKS for i,s in enumerate(SEEDS)])
wb.save(OUT/'消融数据与配对比较.xlsx')

# Build the Markdown master and browser-readable HTML.
md='\n\n<!-- PAGE -->\n\n'.join(p['body'] for p in pages)
(OUT/(STEM+'.md')).write_text(md)
html_pages=[markdown.markdown(p['body'],extensions=['tables']) for p in pages]

# Embed the original TTC fonts first. CID-CFF fonts pre-subset as standalone OTF
# can preserve extractable Unicode while dropping visible CJK glyphs in MuPDF.
# Subset only after the PDF has its own CID mapping, as in the validated short report.
for weight in ['Regular','Bold']:
 shutil.copy2(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc',ASSETS/f'CJK-{weight}.ttc')

css='''
@font-face {font-family:cjk;src:url("assets/CJK-Regular.ttc");}
@font-face {font-family:cjk;src:url("assets/CJK-Bold.ttc");font-weight:bold;}
body {font-family:cjk;font-size:10.2pt;line-height:1.45;color:#26374a;}
h1 {font-size:22pt;color:#173e57;margin:0 0 12pt;}
h2 {font-size:15pt;color:#173e57;margin:0 0 12pt;}
h3 {font-size:11.4pt;color:#216e7b;margin:10pt 0 6pt;}
p {margin:7pt 0;}
table {width:100%;border-collapse:collapse;font-size:8.6pt;margin:10pt 0;}
th {background-color:#e6eff5;color:#163e56;}
th,td {border:.45pt solid #cedae3;padding:4pt;vertical-align:top;}
li {margin:5pt 0;}
blockquote {background-color:#eef5f7;border-left:3pt solid #39788a;padding:8pt;margin:10pt 0;}
img {width:510pt;}
code {font-family:cjk;font-size:8.5pt;}
'''
compact='body {font-size:9.3pt;line-height:1.35;} h2 {font-size:14pt;margin-bottom:9pt;} table {font-size:8pt;margin:7pt 0;} th,td {padding:3pt;} p {margin:6pt 0;}'
appendix_css='body {font-size:8.5pt;line-height:1.24;} h2 {font-size:14pt;} table {font-size:7.2pt;margin:7pt 0;} th,td {padding:2.8pt;} p {margin:5pt 0;}'
browser='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+STEM+'</title><style>'+css+'section{max-width:800px;margin:25px auto;padding:35px;box-shadow:0 1px 8px #ccc;} @media print{section{page-break-before:always;box-shadow:none;}}</style><body>'+''.join('<section>'+h+'</section>' for h in html_pages)+'</body></html>'
(OUT/(STEM+'.html')).write_text(browser)

def break_paths(content):
 parts=re.split('(<[^>]+>)',content)
 return ''.join(re.sub(r'(?<=[A-Za-z0-9_\u4e00-\u9fff])/', '/\u200b',x) if i%2==0 else x for i,x in enumerate(parts))

print('Built content, workbook and figures; rendering PDF...',flush=True)
with tempfile.TemporaryDirectory(prefix='detailed_review_') as tmp:
 raw=Path(tmp)/'raw.pdf';writer=fitz.DocumentWriter(str(raw));media=fitz.paper_rect('a4');toc=[];layout=[];num=0
 for index,(meta,content) in enumerate(zip(pages,html_pages)):
  c=css+(compact if meta['compact'] else '')+(appendix_css if index in [20,21,22] else '')
  if index==19:content=break_paths(content)
  story=fitz.Story(html='<body>'+content+'</body>',user_css=c,archive=fitz.Archive(str(OUT)))
  toc.append([1,meta['title'],num+1]);continuations=0
  while True:
   more,filled=story.place(fitz.Rect(39,42,media.width-39,media.height-43))
   dev=writer.begin_page(media);story.draw(dev);writer.end_page();num+=1;continuations+=1
   layout.append(dict(page=num,section=meta['title'],filled=list(filled),continued=bool(more)))
   if not more:break
   if continuations>=5:raise RuntimeError('Unexpected section overflow: '+meta['title'])
 writer.close()
 with fitz.open(raw) as doc:
  for i,p in enumerate(doc):
   p.draw_line((39,809),(media.width-39,809),color=(.78,.84,.88),width=.5)
   p.insert_text((39,824),'SpecCon-LoRA | Detailed ablation review | 2026-10-25',fontsize=7.5,color=(.34,.43,.50))
   p.insert_text((media.width-65,824),f'{i+1} / {len(doc)}',fontsize=7.5,color=(.34,.43,.50))
  doc.set_toc(toc);doc.set_metadata(dict(title=STEM,author='SpecCon-LoRA experiment review',subject='Detailed ablations, paired evidence, limitations and manuscript revision'))
  print('Embedding and subsetting CJK fonts in the completed PDF...',flush=True)
  doc.subset_fonts(fallback=True)
  doc.save(OUT/(STEM+'.pdf'),garbage=4,clean=True,deflate=True,deflate_fonts=True,use_objstms=0)

# Editable Word version, preserving real tables and inline emphasis.
word=Document();section=word.sections[0];section.page_height=Inches(11.69);section.page_width=Inches(8.27);section.top_margin=Inches(.65);section.bottom_margin=Inches(.65);section.left_margin=section.right_margin=Inches(.62)
for sty in ['Normal','Heading 1','Heading 2','Heading 3']:
 style=word.styles[sty];style.font.name='Noto Sans CJK SC';style._element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC');style.font.size=Pt(10 if sty=='Normal' else 14)
def inline(par,node,bold=False,italic=False):
 if node.text:par.add_run(node.text).bold=bold
 for child in node:
  if child.tag=='img':par.add_run().add_picture(str(OUT/child.attrib['src']),width=Inches(6.8))
  else:inline(par,child,bold or child.tag in ['strong','b'],italic or child.tag in ['em','i'])
  if child.tail:par.add_run(child.tail).bold=bold
def add_node(node):
 if node.tag in ['h1','h2','h3']:
  par=word.add_heading(level=min(int(node.tag[1]),3));inline(par,node)
 elif node.tag=='table':
  rs=node.findall('.//tr');cols=max(len(list(r)) for r in rs);tab=word.add_table(rows=0,cols=cols);tab.style='Table Grid'
  for ri,r in enumerate(rs):
   cells=tab.add_row().cells
   for ci,cell in enumerate(r):
    inline(cells[ci].paragraphs[0],cell,bold=ri==0)
    for run in cells[ci].paragraphs[0].runs:run.font.size=Pt(8)
 elif node.tag in ['ol','ul']:
  for child in node:par=word.add_paragraph(style='List Number' if node.tag=='ol' else 'List Bullet');inline(par,child)
 elif node.tag=='blockquote':
  for child in node:par=word.add_paragraph(style='Quote');inline(par,child)
 else:par=word.add_paragraph();inline(par,node)
for i,h in enumerate(html_pages):
 if i:word.add_page_break()
 for node in ET.fromstring('<div>'+h+'</div>'):add_node(node)
word.save(OUT/(STEM+'.docx'))

# Validation: independent render, readable CJK, page geometry and numeric anchors.
pdf=OUT/(STEM+'.pdf');reader=pypdf.PdfReader(pdf,strict=True)
previews=OUT/'previews';previews.mkdir(exist_ok=True)
with fitz.open(pdf) as doc:
 texts=[p.get_text() for p in doc];fulltext='\n'.join(texts)
 for term in ['219/219','2.529','1.471','0.481','0.298','0.784','0.361','独立校准','论文','seed42']:
  assert term in re.sub(r'\s+', '', unicodedata.normalize('NFKC',fulltext)),term
 assert '\ufffd' not in fulltext
 assert len(reader.pages)==len(doc)
 assert all(len(t)>90 for t in texts)
 outside=[];cjk_visibility=[]
 for i,p in enumerate(doc):
  for block in p.get_text('blocks'):
   if block[0]<-1 or block[1]<-1 or block[2]>p.rect.width+1 or block[3]>p.rect.height+1:outside.append([i+1,block[:4]])
  pix=p.get_pixmap(matrix=fitz.Matrix(1.0,1.0));pix.save(previews/f'page_{i+1:02}.png')
  raster=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
  candidates=[]
  for b in p.get_text('rawdict')['blocks']:
   if 'lines' not in b:continue
   for line in b['lines']:
    for span in line['spans']:
     for ch in span['chars']:
      if '\u4e00'<=ch['c']<='\u9fff':candidates.append(ch)
  samples=candidates[::max(1,len(candidates)//40)][:40];visible=0
  for ch in samples:
   x0,y0,x1,y1=ch['bbox'];crop=raster.crop((max(0,int(x0)),max(0,int(y0)),min(pix.width,math.ceil(x1)),min(pix.height,math.ceil(y1))))
   if crop.width and crop.height and np.any(np.asarray(crop).min(axis=2)<180):visible+=1
  assert samples and visible/len(samples)>.9,(i+1,'CJK glyphs are not visibly rendered',visible,len(samples))
  cjk_visibility.append(dict(page=i+1,sampled=len(samples),visible=visible))
 assert not outside,outside
 # Compact visual contact sheets, six pages per image.
 for start in range(0,len(doc),6):
  contact=Image.new('RGB',(1020,1450),'#e5eaf0')
  for j in range(min(6,len(doc)-start)):
   im=Image.open(previews/f'page_{start+j+1:02}.png').convert('RGB');im.thumbnail((320,675))
   x=15+(j%3)*337;y=15+(j//3)*720;contact.paste(im,(x,y));ImageDraw.Draw(contact).text((x,y+im.height+6),f'Page {start+j+1}',fill='black')
  contact.save(previews/f'contact_{start//6+1}.png')
 page_count=len(doc)
with pdfium.PdfDocument(str(pdf)) as check:
 assert len(check)==page_count
 for p in check:
  bm=p.render(scale=.4);assert bm.width>0 and bm.height>0;bm.close();p.close()
assert len(Document(OUT/(STEM+'.docx')).tables)>=30
validation=dict(created_at=STAMP,pdf=str(pdf),pages=page_count,bytes=pdf.stat().st_size,sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),sections=len(pages),layout=layout,character_counts=[len(t) for t in texts],source_statistics_recomputed=True,strict_parse=True,cjk_text_extraction=True,cjk_visible_glyph_checks=cjk_visibility,pdfium_all_pages_rendered=True,no_out_of_page_blocks=True)
(OUT/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(snapshot_at=STAMP,sources=source_records),ensure_ascii=False,indent=2)+'\n')
(OUT/'README.md').write_text(f'# SpecCon-LoRA 详细消融报告\n\n快照：{STAMP}\n\n- {STEM}.pdf：{page_count} 页详细报告。\n- {STEM}.docx / .md：可编辑正文。\n- 消融数据与配对比较.xlsx：完整数据与来源。\n- 三份“可投稿图.pdf”：独立矢量图。\n- sources/：冻结输入；SOURCE_MANIFEST.json：来源与哈希。\n- VALIDATION.json：数据与PDF检查。\n\n普通重复为三训练种子；校准子集为固定seed42的五个可能重叠子集。未完成HPO仅保留读取时快照。\n')
print(json.dumps({k:v for k,v in validation.items() if k not in ['layout','character_counts']},ensure_ascii=False,indent=2),flush=True)
print('section_page_counts',json.dumps({x['section']:sum(y['section']==x['section'] for y in layout) for x in layout},ensure_ascii=False),flush=True)
