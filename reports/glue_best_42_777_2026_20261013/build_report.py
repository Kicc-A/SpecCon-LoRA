import glob
import json
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path('/workspace/SpecCon-LoRA')
EXP = ROOT / 'experiments'
OUT = Path(__file__).resolve().parent
ORDER = ('cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb')
SEEDS = (42, 777, 2026)
PAPER = {
    'cola': (63.59, 1.13), 'mnli': (87.48, 0.13),
    'mrpc': (89.22, 0.85), 'qnli': (93.01, 0.21),
    'qqp': (90.85, 0.05), 'rte': (80.02, 0.91),
    'sst2': (94.61, 0.57), 'stsb': (90.17, 0.20),
}
PAPER_URL = 'https://aclanthology.org/2026.findings-acl.1693.pdf'
BASE_SUMMARY = EXP / 'glue_rdrop_best_seed2026_gpu01_20261008_v1/summary.json'
SPECIAL = {
    'mrpc': ('small_glue_rdrop_dropout_labelsmoothing_3seeds_gpu23_20261008_v1', 'mrpc_ls005'),
    'qnli': ('qnli_sst2_lambda_sweep_20261012_v1', 'qnli/lambda0p0'),
    'rte': ('rte_speccon_hparam_sweep_63ep_20261011_v1', 'lambda05'),
    'sst2': ('qnli_sst2_lambda2_20261013_v1', 'sst2/lambda2p0'),
}
LABELS = {'cola': 'CoLA', 'mnli': 'MNLI', 'mrpc': 'MRPC', 'qnli': 'QNLI',
          'qqp': 'QQP', 'rte': 'RTE', 'sst2': 'SST-2', 'stsb': 'STS-B'}
METRICS = {'cola': 'MCC', 'mnli': 'm/mm Acc', 'mrpc': 'Acc', 'qnli': 'Acc',
           'qqp': 'Acc', 'rte': 'Acc', 'sst2': 'Acc', 'stsb': 'Pearson'}
FONT = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
FONT_BOLD = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')


def score(task, record):
    splits = record['splits']
    if task == 'mnli':
        return 50 * (splits['matched']['accuracy'] + splits['mismatched']['accuracy'])
    metric = 'matthews_correlation' if task == 'cola' else 'pearson' if task == 'stsb' else 'accuracy'
    return 100 * splits['validation'][metric]


def load_data():
    baseline = json.loads(BASE_SUMMARY.read_text())
    base = {g['task']: g for g in baseline['groups']}
    result = {}
    for task in ORDER:
        if task in SPECIAL:
            campaign, group = SPECIAL[task]
            sources = [EXP / campaign / 'runs' / group / f'seed{seed}' for seed in SEEDS]
            label = {'mrpc': 'label smoothing 0.05', 'qnli': 'lambda 0',
                     'rte': 'lambda 0.5', 'sst2': 'lambda 2, 38 epochs'}[task]
        else:
            sources = [Path(x['source']) for x in sorted(base[task]['per_seed'], key=lambda x: x['seed'])]
            label = 'base configuration'
        runs = []
        for seed, source in zip(SEEDS, sources):
            m = json.loads((source / 'metrics.json').read_text())
            assert m['state'] == 'complete' and m['task'] == task and m['seed'] == seed
            assert m['final']['epoch'] == m['epochs']
            runs.append({'seed': seed, 'score': score(task, m['final']),
                         'source': str(source.relative_to(ROOT)),
                         'hyperparameters': m['hyperparameters'], 'method': m['method'],
                         'adapter_parameters': m['adapter_parameters']})
        # Each task uses one configuration for all three seeds; only seed-specific RNG references may differ.
        def canonical(hp):
            return {k: v for k, v in hp.items() if k != 'training_rng_reference'}
        assert all(canonical(r['hyperparameters']) == canonical(runs[0]['hyperparameters']) for r in runs)
        assert all(r['adapter_parameters'] == runs[0]['adapter_parameters'] for r in runs)
        vals = [r['score'] for r in runs]
        result[task] = {'label': label, 'runs': runs, 'mean': statistics.mean(vals),
                        'sample_sd': statistics.stdev(vals), 'paper_mean': PAPER[task][0],
                        'paper_sd': PAPER[task][1], 'delta': statistics.mean(vals) - PAPER[task][0],
                        'hyperparameters': canonical(runs[0]['hyperparameters']),
                        'adapter_parameters': runs[0]['adapter_parameters']}
    return result


def pending_status():
    directory = EXP / 'qnli_sst2_lambda2_20261013_v1'
    complete = []
    for p in directory.glob('runs/**/metrics.json'):
        m = json.loads(p.read_text())
        if m.get('state') == 'complete':
            complete.append(f"{m['task']} seed{m['seed']}")
    return sorted(complete)


def text_at(ax, x, y, s, size=10, bold=False, ha='left', color='#18334d'):
    ax.text(x, y, s, fontproperties=FONT_BOLD if bold else FONT,
            fontsize=size, ha=ha, va='center', color=color)


def line(ax, y, lw=0.8, color='#41566b'):
    ax.plot([0.035, 0.965], [y, y], color=color, lw=lw, transform=ax.transAxes)


def setup_page(title, subtitle):
    fig = plt.figure(figsize=(16.5, 9.2), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    text_at(ax, 0.035, 0.935, title, size=21, bold=True)
    text_at(ax, 0.035, 0.888, subtitle, size=10, color='#587084')
    line(ax, 0.865, 1.3)
    return fig, ax


def page_one(data, snapshot):
    fig, ax = setup_page('GLUE 最优完整三 seed 结果  |  DPLoRA 格式对照',
                         'RoBERTa-base · seed 42 / 777 / 2026 · 最终轮验证集 · 分数均为百分制，越高越好')
    xs = [0.038, 0.265] + [0.335 + i * 0.0775 for i in range(8)] + [0.965]
    headers = ['Method', 'Params\n(M)'] + [LABELS[t] for t in ORDER] + ['Avg.']
    for i, h in enumerate(headers):
        text_at(ax, xs[i], 0.815, h, 9.4, bold=True, ha='left' if i == 0 else 'right')
    line(ax, 0.784, 1.1)
    paper_avg = statistics.mean(PAPER[t][0] for t in ORDER)
    our_avg = statistics.mean(data[t]['mean'] for t in ORDER)
    rows = [('DPLoRA (p = 0.4)', '0.7', [f'{PAPER[t][0]:.2f}' for t in ORDER], f'{paper_avg:.2f}')]
    for idx, seed in enumerate(SEEDS):
        vals = [data[t]['runs'][idx]['score'] for t in ORDER]
        rows.append((f'SpecCon-LoRA + R-Drop  ·  seed {seed}', '0.756',
                     [f'{v:.2f}' for v in vals], f'{statistics.mean(vals):.2f}'))
    rows.append(('SpecCon-LoRA + R-Drop  ·  mean ± SD', '0.756',
                 [f"{data[t]['mean']:.2f} ± {data[t]['sample_sd']:.2f}" for t in ORDER],
                 f'{our_avg:.2f}'))
    rows.append(('Δ vs DPLoRA (mean)', '—', [f"{data[t]['delta']:+.2f}" for t in ORDER],
                 f'{our_avg - paper_avg:+.2f}'))
    ys = [0.74, 0.67, 0.60, 0.53, 0.44, 0.36]
    ax.axhspan(0.404, 0.477, xmin=0.035, xmax=0.965, color='#eaf2f8', zorder=0)
    for j, (name, params, vals, avg) in enumerate(rows):
        bold = j in (4, 5)
        text_at(ax, xs[0], ys[j], name, 9.25, bold=bold)
        text_at(ax, xs[1], ys[j], params, 9.2, bold=bold, ha='right')
        for i, value in enumerate(vals):
            text_at(ax, xs[i + 2], ys[j], value, 8.1 if j == 4 else 9.2, bold=bold, ha='right',
                    color='#126455' if j == 5 and value.startswith('+') else '#18334d')
        text_at(ax, xs[-1], ys[j], avg, 9.2, bold=bold, ha='right')
        if j in (0, 3):line(ax, ys[j] - 0.04)
    line(ax, 0.316, 1.3)
    text_at(ax, 0.038, 0.265, '各任务采用同一配置的完整三 seed 最终轮结果；SST-2 均值与原配置持平，本次换入 λ=2。', 9.7)
    text_at(ax, 0.038, 0.23, 'CoLA：MCC；MNLI：matched/mismatched 准确率均值；STS-B：Pearson；其余：accuracy。均乘 100。', 9.7)
    text_at(ax, 0.038, 0.195, 'SD 为本实验三个 seed 的样本标准差；DPLoRA 行的 SD 详见第 2 页。Avg. 为 8 项任务均值。', 9.7)
    text_at(ax, 0.038, 0.16, 'DPLoRA 数值取论文 Table 8 (p = 0.4)；两组实验的调参过程和训练开销未统一。', 9.7)
    text_at(ax, 0.038, 0.083, f'结果快照：{snapshot}  ·  第 1 页 / 2', 9, color='#6a7d8c')
    return fig


def page_two(data, snapshot, pending):
    fig, ax = setup_page('入选配置与核验依据', '每行对应主表中的一个任务；下方列出 DPLoRA 报告值及原始实验目录')
    xs = [0.039, 0.135, 0.247, 0.345, 0.44, 0.535, 0.63, 0.965]
    heads = ['Task', 'Metric', 'Epochs', 'LR', 'λ', 'Dropout', 'LS', '3-seed mean ± SD']
    for i, h in enumerate(heads):
        text_at(ax, xs[i], 0.817, h, 10, bold=True, ha='left' if i < 2 else 'right')
    line(ax, 0.786, 1.1)
    ys = [0.751 - i * 0.056 for i in range(8)]
    for task, y in zip(ORDER, ys):
        g = data[task]; hp = g['hyperparameters']
        cells = [LABELS[task], METRICS[task], str(hp['epochs']), f"{hp['lr']:.2g}",
                 f"{hp['lambda']:g}", f"{hp.get('dropout', 0):g}",
                 f"{hp.get('label_smoothing', 0):g}", f"{g['mean']:.2f} ± {g['sample_sd']:.2f}"]
        for i, cell in enumerate(cells):
            text_at(ax, xs[i], y, cell, 9.4, bold=(i == 0), ha='left' if i < 2 else 'right')
    line(ax, 0.28, 1.1)
    text_at(ax, 0.039, 0.247, '实验目录（相对 /workspace/SpecCon-LoRA/experiments/）：', 9.2, bold=True)
    for i, task in enumerate(ORDER):
        paths = [r['source'] for r in data[task]['runs']]
        campaigns = sorted(set(Path(p).parts[1] for p in paths))
        text_at(ax, 0.039, 0.22 - i * 0.019,
                f"{LABELS[task]}: {', '.join(campaigns)}", 6.9)
    text_at(ax, 0.039, 0.055, 'DPLoRA Table 8 的 SD：' + '  ·  '.join(f"{LABELS[t]} {PAPER[t][1]:.2f}" for t in ORDER), 8.2)
    text_at(ax, 0.039, 0.026,
            f"λ=2：SST-2 已纳入；QNLI 完成 {sum(item.startswith('qnli ') for item in pending)}/3，主表仍用 λ=0。来源：{PAPER_URL}", 8.2)
    text_at(ax, 0.965, 0.026, '第 2 页 / 2', 8.5, ha='right', color='#6a7d8c')
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    data = load_data()
    pending = pending_status()
    paper_avg = statistics.mean(PAPER[t][0] for t in ORDER)
    our_avg = statistics.mean(data[t]['mean'] for t in ORDER)
    result = {'snapshot': snapshot, 'seeds': SEEDS, 'selection_rule': 'highest complete three-seed mean of final-epoch validation metric for each task',
              'paper_source': PAPER_URL, 'paper_avg': paper_avg, 'our_avg': our_avg,
              'delta_avg': our_avg - paper_avg, 'lambda2_completed': pending,
              'selection_notes': {'sst2': 'User-requested update to the completed lambda=2.0, 38-epoch campaign; its three-seed final mean ties the previously selected 15-epoch lambda=1.5 campaign.'},
              'tasks': data}
    (OUT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    pdf = OUT / 'SpecCon-LoRA_GLUE_best_42_777_2026_vs_DPLoRA.pdf'
    figures = [page_one(data, snapshot), page_two(data, snapshot, pending)]
    with PdfPages(pdf) as doc:
        for fig in figures:doc.savefig(fig, facecolor='white')
    for i, fig in enumerate(figures, 1):
        fig.savefig(OUT / f'preview_page{i}.png', dpi=120, facecolor='white')
        plt.close(fig)
    print(json.dumps({'pdf': str(pdf), 'our_avg': our_avg, 'paper_avg': paper_avg,
                      'delta': our_avg - paper_avg, 'pending': pending}, ensure_ascii=False))


if __name__ == '__main__':
    main()
