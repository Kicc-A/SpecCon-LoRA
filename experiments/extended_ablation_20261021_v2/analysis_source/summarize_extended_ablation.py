#!/usr/bin/env python3
"""Generate paired final-epoch tables as the frozen experiment matrix completes."""
import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time


def load(path):
    return json.loads(Path(path).read_text())


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def save_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    source = root / 'results.json'
    if not source.exists():
        return
    results = load(source)
    runs, refs = results['runs'], results['references']
    lookup = {(r['task'], r['name'], r['seed']): r for r in runs + refs}
    grouped = collections.defaultdict(list)
    for r in runs + refs:
        grouped[r['task'], r['name']].append(r)
    summary = [dict(task=task, name=name, **describe([r['score'] for r in rows]))
               for (task, name), rows in sorted(grouped.items())]
    contrasts = []

    def pair(task, name, left, right, expected=3, group='training_seed'):
        delta, details = [], []
        for seed in [42, 777, 2026]:
            a, b = lookup.get((task, left, seed)), lookup.get((task, right, seed))
            if a and b:
                delta.append(a['score'] - b['score'])
                details.append(dict(seed=seed, left=a['score'], right=b['score'], difference=delta[-1]))
        contrasts.append(dict(task=task, comparison=name, left=left, right=right,
            expected_n=expected, complete=len(delta) == expected, unit=group,
            **describe(delta), paired_values=details))

    for task in ['cola', 'mrpc', 'rte']:
        pair(task, 'allocation_same_budget_svd', 'full', 'near_uniform_svd')
        pair(task, 'allocation_same_budget_random', 'adaptive_random', 'near_uniform_random')
        pair(task, 'direction_same_gram', 'full', 'orthogonal_svdnorm')
        pair(task, 'init_same_frobenius', 'full', 'random_svdnorm')
        pair(task, 'svd_scale_only', 'full', 'svd_randomnorm')
        pair(task, 'random_scale_only', 'random_svdnorm', 'adaptive_random')
        pair(task, 'speccon_KL_same_compute', 'full', 'speccon_two_ce')
        pair(task, 'uniform_KL_same_compute', 'uniform8_random', 'uniform8_two_ce')
        for name in ['H2', 'H8', 'H16', 'N128', 'N256', 'N1024', 'gamma0.7', 'gamma1.0', 'gamma1.5']:
            pair(task, name + '_minus_default', name, 'full')
        for rank in [3, 6, 8]:
            pair(task, f'budget_rank{rank}_allocation', f'budget_r{rank}_adaptive',
                 'uniform8_svd' if rank == 8 else f'budget_r{rank}_uniform')
        subset = [r for r in runs if r['task'] == task and r['group'] == 'calibration_subset']
        differences, details = [], []
        for rep in range(5):
            a = next((r for r in subset if r['replicate'] == rep and r['consistency_lambda'] > 0), None)
            b = next((r for r in subset if r['replicate'] == rep and r['consistency_lambda'] == 0), None)
            if a and b:
                differences.append(a['score'] - b['score'])
                details.append(dict(replicate=rep, seed=42, left=a['score'], right=b['score'], difference=differences[-1]))
        contrasts.append(dict(task=task, comparison='independent_subset_full_minus_lambda0',
            left='full', right='lambda0', expected_n=5, complete=len(differences) == 5,
            unit='calibration_subset_fixed_training_seed42', **describe(differences), paired_values=details))

    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    payload = dict(updated=datetime.now().astimezone().isoformat(), completed=results['completed'],
        total=results['total'], results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        note='Descriptive paired differences only; incomplete groups are not final conclusions. '
             'Calibration replicates overlap and share training seed42.',
        summaries=summary, contrasts=contrasts)
    temp = output / 'paired_results.json.tmp'
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temp.replace(output / 'paired_results.json')
    save_csv(output / 'runs.csv', runs, ['id', 'task', 'group', 'name', 'seed', 'replicate', 'budget', 'score',
        'elapsed_seconds', 'peak_memory_mib', 'output'])
    save_csv(output / 'summary.csv', summary, ['task', 'name', 'n', 'mean', 'sample_std'])
    save_csv(output / 'paired_contrasts.csv', contrasts,
        ['task', 'comparison', 'left', 'right', 'unit', 'n', 'expected_n', 'complete', 'mean', 'sample_std'])
    manifest = load(root / 'manifest.json')
    completed_groups = collections.Counter(row['group'] for row in runs)
    group_titles = {'baseline': '复用参考', 'equal_budget': '等参数预算',
        'initialization': '初始化方向与尺度', 'rdrop_equal_compute': '等计算R-Drop',
        'blocks': '分块数H', 'calibration_size': '校准样本数N', 'gamma': 'gamma敏感性',
        'budget_curve': '预算曲线', 'calibration_subset': '独立校准子集下游训练'}
    lines = ['# 新增消融：完整进度、配置结果与配对比较', '',
        f"完成 {results['completed']}/{results['total']}。均值和标准差仅为描述性统计；差值方向为左项减右项。",
        '这里的完成数指独立正式训练次数：一个配置×一个任务×一个种子算一次。复用参考和短预检不计新增训练。',
        '普通对照按相同训练种子配对；独立校准按子集配对，固定训练seed42，不能混作5训练种子。', '',
        '| 实验组 | 完成/计划训练数 |', '| --- | ---: |']
    for group, expected in manifest['groups'].items():
        lines.append(f"| {group_titles[group]} | {completed_groups[group]}/{expected} |")
    lines += ['', 'CoLA为MCC×100，MRPC/RTE为Accuracy×100。±为样本标准差（ddof=1）；普通配置预期n=3，未满3明确标注，不能当完整三种子结果。',
        'near_uniform为attention rank4、FFN rank5，总预算755712；uniform8为1327104参数，跨预算行不能当严格等预算对照。']
    by_group = collections.defaultdict(list)
    for row in runs + refs:
        if row['name'] not in by_group[row['group']]:
            by_group[row['group']].append(row['name'])
    for group in group_titles:
        lines += ['', '## ' + group_titles[group], '']
        if not by_group[group]:
            lines.append('尚无通过完整末轮核验的新增结果。')
            continue
        lines += ['| 配置 | CoLA | MRPC | RTE |', '| --- | ---: | ---: | ---: |']
        for name in by_group[group]:
            values = []
            for task in ['cola', 'mrpc', 'rte']:
                item = next((s for s in summary if s['task'] == task and s['name'] == name), None)
                if item is None:
                    values.append('—')
                elif item['sample_std'] is None:
                    values.append(f"{item['mean']:.3f}（n={item['n']}）")
                else:
                    note = '' if item['n'] == 3 else f"（n={item['n']}/3）"
                    values.append(f"{item['mean']:.3f} ± {item['sample_std']:.3f}{note}")
            lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 完整配对比较', '',
        '| 任务 | 比较 | 已完成配对 | 差值均值 | 样本标准差 |', '| --- | --- | ---: | ---: | ---: |']
    for row in contrasts:
        if not row['n']:
            continue
        sd = '—' if row['sample_std'] is None else f"{row['sample_std']:.3f}"
        lines.append(f"| {row['task']} | {row['comparison']} | {row['n']}/{row['expected_n']} | {row['mean']:+.3f} | {sd} |")
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(payload['updated'], 'analyzed', results['completed'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    previous = None
    while True:
        source = args.root / 'results.json'
        stamp = source.stat().st_mtime_ns if source.exists() else None
        if stamp != previous:
            analyze(args.root)
            previous = stamp
        if not args.watch:
            return
        status = load(args.root / 'status.json')
        if status['state'] in ['complete', 'complete_with_failures', 'preflight_failed', 'stopped']:
            analyze(args.root)
            return
        pid = status.get('controller_pid')
        if pid and not Path(f'/proc/{pid}').exists():
            print('Controller is no longer running; see controller.log. No experiments restarted.', flush=True)
            return
        time.sleep(45)


if __name__ == '__main__':
    main()
