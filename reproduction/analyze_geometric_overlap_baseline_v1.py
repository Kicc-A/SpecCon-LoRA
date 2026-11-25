#!/usr/bin/env python3
"""Compare observed A/B left-subspace recurrence with isotropic random overlap."""
import csv
from pathlib import Path

import numpy as np

SOURCE=Path('/workspace/SpecCon-LoRA/reports/independent_calibration_mechanism_20261019/candidate_metrics.csv')
ROOT=Path('/workspace/SpecCon-LoRA/experiments/geometric_predictivity_validation_v1')


def main():
    values={}
    with SOURCE.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['order']!='0' or int(row['k'])<3:
                continue
            dimension=int(row['shape'].split('x')[0])
            k=int(row['k'])
            baseline=k/dimension
            key=(row['task'],row['module_type'],row['shape'],k)
            values.setdefault(key,[]).append((float(row['recurrence']),baseline))
    rows=[]
    for (task,kind,shape,k),samples in sorted(values.items()):
        observed=float(np.mean([s[0] for s in samples]))
        baseline=samples[0][1]
        assert all(s[1]==baseline for s in samples)
        rows.append({'task':task,'module_type':kind,'shape':shape,'k':k,'n':len(samples),
            'observed_recurrence_mean':observed,
            'isotropic_random_expectation':baseline,
            'observed_minus_random':observed-baseline})
    with (ROOT/'random_overlap_baseline.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=ROOT/'REPORT.md'
    body=report.read_text()
    marker='## 可能机制解释'
    note=('### 子空间维度的随机重合基线\n\n'
        '原始记录保留每个候选的输出维度 d、rank k 与 A/B 左子空间重现度。'
        '若两个 k 维子空间在 d 维输出空间独立且各向同性，'
        '归一化投影重合的期望为 k/d。'
        '按任务、模块类型、形状和 k 汇总的实测值与该解析零基线见 '
        '`random_overlap_baseline.csv`。'
        '这只是维度造成的随机重合基线，不是将原梯度随机化后的新实测实验。'
        '在固定 k 和模块类型内，该基线为常数，因此不能单独解释控制后的相关性。\n\n')
    if 'random_overlap_baseline.csv' not in body:
        report.write_text(body.replace(marker,note+marker))
    print('random overlap baseline rows',len(rows),'minimum excess',min(r['observed_minus_random'] for r in rows))


if __name__=='__main__':
    main()
