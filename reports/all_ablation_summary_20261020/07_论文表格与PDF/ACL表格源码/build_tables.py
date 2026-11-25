"""Build ACL booktabs, an Excel workbook, and Markdown from the frozen snapshot."""
import csv
import json
import re
import statistics as stats
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter

OUT = Path(__file__).resolve().parent
D = json.loads((OUT / 'snapshot.json').read_text())
TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']
LABELS = dict(zip(TASKS, ['CoLA', 'MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B']))
SEEDS = [42, 777, 2026]
TABLES = []
MISSING = '—'


def num(value, digits=2):
    return MISSING if value is None else f'{value:.{digits}f}'


def avg(values, digits=2):
    if not values:
        return MISSING
    if len(values) == 1:
        return num(values[0], digits)
    return f'{stats.mean(values):.{digits}f} ± {stats.stdev(values):.{digits}f}'


def add(name, title, headers, rows, caption, align=None, breaks=()):
    assert all(len(row) == len(headers) for row in rows), name
    TABLES.append({'id': name, 'title': title, 'headers': headers, 'rows': rows,
                   'caption': caption, 'align': align or ('l' + 'r' * (len(headers) - 1)),
                   'breaks': list(breaks)})


def selected(task, seed=42):
    return next(x for x in D['selected'][task] if x['seed'] == seed)


def main_runs(task):
    if task != 'qnli':
        return D['selected'][task]
    group = next(x for x in D['sensitivity'] if x['task'] == task and x['lambda'] == 0.7)
    assert sorted(x['seed'] for x in group['results']) == SEEDS
    assert all(x['hyperparameters']['lambda'] == 0.7 for x in group['results'])
    return group['results']


def main_selected(task, seed=42):
    return next(x for x in main_runs(task) if x['seed'] == seed)


def component(task, variant, seed=42):
    return next(x['result'] for x in D['components'] if (x['task'], x['variant'], x['seed']) == (task, variant, seed))


def value(result):
    return None if result is None else result['score']


def main_results():
    rows = []
    for task in TASKS:
        group = D['selected_report']['tasks'][task]
        scores = [r['score'] for r in main_runs(task)]
        mean = stats.mean(scores)
        paper_score = f"{group['paper_mean']:.2f} ± {group['paper_sd']:.2f}"
        ours_score = avg(scores)
        if mean >= group['paper_mean']:
            ours_score = '**' + ours_score + '**'
        if group['paper_mean'] >= mean:
            paper_score = '**' + paper_score + '**'
        metric = {'cola': 'MCC', 'mnli': 'Acc. (m/mm)', 'stsb': 'Pearson'}.get(task, 'Accuracy')
        rows.append([LABELS[task], metric, paper_score, ours_score, f"{mean-group['paper_mean']:+.2f}"])
    paper = stats.mean(D['selected_report']['tasks'][t]['paper_mean'] for t in TASKS)
    ours = stats.mean(stats.mean(r['score'] for r in main_runs(t)) for t in TASKS)
    rows.append(['Macro average', '', num(paper), '**' + num(ours) + '**', f'{ours-paper:+.2f}'])
    add('01_glue_main', 'GLUE 主结果', ['Task', 'Metric', 'DPLoRA (paper)', 'SpecCon-LoRA', 'Δ'], rows,
        'GLUE development-set results on RoBERTa-base (all metrics ×100; higher is better). '
        'DPLoRA p=0.4 results are reported by Park et al. (2026), Table 8. SpecCon-LoRA reports mean ± sample SD '
        'over seeds 42/777/2026, using the selected configuration and final epoch for each task; SpecCon adapters have '
        '755,712 parameters. QNLI now uses λ=0.7; λ=0 remains an ablation reference. MRPC uses accuracy, '
        'not the accuracy/F1 composite. Δ is SpecCon minus the paper reference; bold indicates the larger mean. '
        'Configurations were selected on validation results, and the training recipes differ; this is a historical '
        'comparison, not a controlled causal test. The macro average has no pooled SD.', align='llrrr', breaks=[8])


def components():
    definitions = {
        'A': ('Uniform r=8', 'Random', 1327104),
        'B': ('Uniform r=8', 'Grad. SVD', 1327104),
        'C': ('Block spectrum, λ=0', 'Random', 755712),
        'D': ('Block spectrum, λ=0', 'Grad. SVD', 755712),
        'E': ('Spectrum + consistency', 'Random', 755712),
        'F': ('Spectrum + consistency', 'Grad. SVD', 755712),
    }
    rows = []
    for v, (allocation, init, params) in definitions.items():
        rows.append([v, allocation, init, f'{params/1e6:.3f}'] +
                    [num(value(component(t, v))) for t in ['cola', 'mrpc', 'rte']])
    add('02_components_seed42', '分配与初始化消融（seed 42）',
        ['ID', 'Rank allocation', 'A initialization', 'Params (M)', 'CoLA', 'MRPC', 'RTE'], rows,
        'Allocation/initialization ablation at seed 42; fixed final-epoch CoLA MCC and MRPC/RTE accuracy (×100). '
        'Random means Kaiming A; Grad. SVD means gradient-SVD A. B is initialized to zero in all rows. '
        'For E/F, λ is 1.5/1.5/0.5 on CoLA/MRPC/RTE. All rows use R-Drop α=1 and the same within-task recipe '
        '(MRPC label smoothing=0.05). A/B use 1,327,104 parameters, whereas C–F use 755,712; comparisons across '
        'these budget families are not budget controlled. — indicates no completed final result at the snapshot.',
        align='lllrrrr', breaks=[2])
    contrasts = [('F−D', 'Consistency; SVD init', 'F', 'D'),
                 ('E−C', 'Consistency; random init', 'E', 'C'),
                 ('B−A', 'SVD init; uniform ranks', 'B', 'A'),
                 ('D−C', 'SVD init; spectrum ranks', 'D', 'C'),
                 ('F−E', 'SVD init; full allocation', 'F', 'E')]
    rows = []
    for name, purpose, a, b in contrasts:
        deltas = []
        for task in ['cola', 'mrpc', 'rte']:
            x, y = component(task, a), component(task, b)
            deltas.append(MISSING if not x or not y else f"{x['score']-y['score']:+.2f}")
        rows.append([name, purpose] + deltas)
    add('03_paired_contrasts', '同预算组件差值（seed 42）', ['Contrast', 'Factor tested', 'CoLA', 'MRPC', 'RTE'], rows,
        'Paired seed-42 differences in metric points. Each listed contrast preserves the adapter parameter budget '
        'within its allocation family and uses the recorded paired training RNG. Positive values favor the first '
        'variant. Missing operands produce —. These single-seed results do not establish statistical significance; '
        'the cross-budget D−B contrast is deliberately not interpreted as a pure allocation effect.', align='llrrr')


def rdrop():
    rows = []
    for method, label, params in [('uniform', 'Uniform LoRA r=8', 1327104), ('speccon', 'SpecCon-LoRA', 755712)]:
        for alpha in [0, 1]:
            scores = [next(x['result']['score'] for x in D['rdrop42'] if
                          x['task'] == t and x['adapter_method'] == method and x['rdrop_alpha'] == alpha)
                      for t in ['cola', 'mrpc', 'rte']]
            rows.append([label, f'{params/1e6:.3f}', str(alpha), str(1 if alpha == 0 else 2)] + [num(s) for s in scores])
    add('04_rdrop_seed42', 'R-Drop 消融（seed 42）',
        ['Method', 'Params (M)', 'α', 'Forwards', 'CoLA', 'MRPC', 'RTE'], rows,
        'R-Drop recipe ablation, seed 42, fixed final epoch. α=0 uses one forward pass; α=1 uses two forward passes '
        'and symmetric KL. Thus this ablates the complete training recipe rather than the KL term at equal compute. '
        'Within-method budgets are fixed; the two methods have different parameter budgets. MRPC retains label '
        'smoothing=0.05 in all cells. SpecCon λ is 1.5/1.5/0.5 on CoLA/MRPC/RTE. '
        'The completed three-seed extension is reported in Table 14.', align='lrrrrrr', breaks=[2])


def lambda_tables():
    rows = []
    for task in TASKS:
        base = selected(task)
        if task == 'qnli':
            group = next(x for x in D['sensitivity'] if x['task'] == task and x['lambda'] == 1.5)
            base = next((x for x in group['results'] if x['seed'] == 42), None)
        lam = base['hyperparameters']['lambda'] if base else 1.5
        a, b = value(D['lambda0'][task]), value(base)
        rows.append([LABELS[task], f'{lam:g}', num(a, 4), num(b, 4), MISSING if a is None or b is None else f'{b-a:+.4f}'])
        if task == 'qnli':
            alternate = main_selected(task)['score']
            rows.append(['QNLI', '0.7', num(a, 4), num(alternate, 4), f'{alternate-a:+.4f}'])
        if task == 'rte':
            group = next(x for x in D['sensitivity'] if x['task'] == task and x['lambda'] == 1.5)
            alternate = next(x['score'] for x in group['results'] if x['seed'] == 42)
            rows.append(['RTE', '1.5', num(a, 4), num(alternate, 4), f'{alternate-a:+.4f}'])
    add('05_lambda_seed42', '子空间一致性消融（seed 42）',
        ['Task', 'λ', 'Score (λ=0)', 'Score (λ>0)', 'Δ'], rows,
        'Completed consistency-weight ablation, seed 42. Δ is the nonzero-λ score minus λ=0; four decimals '
        'retain small effects. Both settings retain block-wise spectra, SVD initialization and the same task recipe, '
        'including R-Drop α=1. The RTE and QNLI rows each share their task\'s single λ=0 run. '
        'The main table uses RTE λ=0.5 and QNLI λ=0.7. No claim of uniform improvement is implied.', align='lrrrr')
    rows = []
    for group in sorted(D['sensitivity'], key=lambda x: (['rte','qnli'].index(x['task']), x['lambda'])):
        by_seed = {x['seed']: x['score'] for x in group['results']}
        rows.append([LABELS[group['task']], f"{group['lambda']:g}"] + [num(by_seed.get(s), 4) for s in SEEDS] +
                    [str(len(by_seed)), avg(list(by_seed.values()), 4)])
    add('06_lambda_multiseed', 'λ 多种子结果与覆盖情况',
        ['Task', 'λ', 'Seed 42', 'Seed 777', 'Seed 2026', 'n', 'Mean ± SD'], rows,
        'Available final-epoch λ sensitivity results. n is the number of completed seeds, not the planned count. '
        'A single observation has no SD; — is missing, not zero. RTE λ=0 now includes all three completed component-D seeds. '
        'QNLI λ=0/0.7/1.5 and RTE λ=0/0.5/1.5 each have all three completed seeds. QNLI λ=0.7 has a '
        'higher mean than λ=1.5, but a lower mean than λ=0. No statistical significance is asserted.',
        align='lrrrrrr', breaks=[3])


def historical():
    rows = []
    for task in ['cola', 'rte']:
        runs = [x for x in D['dplora_local'] if x['task'] == task]
        by_seed = {x['seed']: x['score'] for x in runs}
        rows.append([LABELS[task], str(runs[0]['epochs'])] + [num(by_seed.get(s)) for s in [42,777,2025]] +
                    [str(len(runs)), avg(list(by_seed.values()))])
    add('07_dplora_local', 'DPLoRA 本地复现', ['Task','Epochs','Seed 42','Seed 777','Seed 2025','n','Mean ± SD'], rows,
        'Completed local DPLoRA p=0.4 runs on RoBERTa-base, separate from the paper-reported reference in Table 1. '
        'CoLA reports final MCC and RTE final accuracy (×100). RTE seed 777 uses the completed restart2 run. '
        'RTE seed 2025 failed; other GLUE tasks have no completed local results. These runs use their own historical '
        'training protocol, including 67 epochs for CoLA. Seeds differ from the SpecCon three-seed set.', align='lrrrrrr')
    rows = [
        ['MetaMathQA', 'GSM8K', '77', 'Accuracy', num(D['math']['accuracy']*100)],
        ['CodeFeedback', 'HumanEval', '42', 'pass@1', num(D['code']['pass@1']*100)],
        ['CodeFeedback', 'HumanEval', '42', 'pass@5', num(D['code']['pass@5']*100)],
    ]
    add('08_nlg_history', '数学与代码历史实验', ['Training data','Evaluation','Seed','Metric','Score (%)'], rows,
        'Retained historical SpecCon-LoRA results on Llama-3.1-8B-Base. These are selected best single runs, not '
        'multi-seed averages. GSM8K: 967/1,319 correct, using the recorded LoRA-GA-style integer-answer extractor. '
        'HumanEval: 164 problems with five samples per problem; pass@1 and pass@5 come from the same run. '
        'NLG uses SVD-based initialization with small nonzero B, whereas GLUE uses zero B.', align='llrlr')
    lookup = {x['指标']: x for x in D['mtbench']}
    rows = []
    for label, key in [('DPLoRA (historical)', 'DPLoRA历史'), ('SpecCon-LoRA', 'SpecCon')]:
        rows.append([label, '42', '2.156'] + [f'{float(lookup[k][key]):.3f}' for k in ['第一轮','第二轮','总分']])
    add('09_mtbench_history', 'MT-Bench 历史对照', ['Method','Seed','Final params (M)','Turn 1','Turn 2','Overall'], rows,
        'Archived MT-Bench scores on Llama-3.2-1B after Alpaca training: 80 questions, 160 turn-level GPT-4 judgments '
        'per method. Both final adapters contain 2,156,032 parameters. DPLoRA starts from a larger training budget, '
        'and initialization, learning rates, regularization and judging dates differ. Scores are preserved as recorded; '
        'this comparison does not isolate rank allocation. Only seed 42 has complete SpecCon judgments.', align='lrrrrr')


def appendices():
    rows = []
    for task in TASKS:
        h = main_selected(task)['hyperparameters']
        rows.append([LABELS[task], str(h['epochs']), f"{h['lambda']:g}", f"{h['lr']:.1e}",
                     f"{h['wd']:.3f}", str(h['max_len']), f"{h['warmup']:.3f}",
                     f"{h['dropout']:g}", f"{h['label_smoothing']:g}"])
    add('10_glue_settings', 'GLUE 主表配置', ['Task','Epochs','λ','LR','WD','Length','Warmup','Dropout','LS'], rows,
        'Frozen SpecCon-LoRA settings for the selected GLUE main results. All runs use 755,712 adapter parameters, '
        'rank range [2,16], γ=1.2, 16 calibration batches split into four blocks, batch size 32, FP32, linear LR '
        'schedule, fixed scaling 2, equal A/B learning rates, and R-Drop-style coefficient 1. STS-B uses regression '
        'consistency. LS denotes label smoothing; dropout denotes LoRA dropout. Training and scheduler epochs match.',
        align='lrrrrrrrr')
    complete = lambda cells: sum(x['result'] is not None for x in cells)
    lambda15 = sum(len(x['results']) for x in D['sensitivity'] if x['lambda'] == 1.5)
    lambda07 = sum(len(x['results']) for x in D['sensitivity'] if x['lambda'] == .7)
    status = lambda count, total: 'Complete' if count == total else 'Incomplete'
    rows = [
        ['Selected GLUE main results', '8 tasks × 3 seeds', '24/24', 'Complete'],
        ['Allocation × initialization', '3 tasks × 6 variants × 3 seeds', f"{complete(D['components'])}/54", status(complete(D['components']), 54)],
        ['R-Drop, seed 42', '3 tasks × 2 methods × 2 settings', '12/12', 'Complete'],
        ['R-Drop, three seeds', '3 tasks × 2 methods × 2 settings × 3', f"{complete(D['extended_rdrop'])}/36", status(complete(D['extended_rdrop']), 36)],
        ['Consistency λ=0', '8 tasks; seed 42', f"{sum(x is not None for x in D['lambda0'].values())}/8", status(sum(x is not None for x in D['lambda0'].values()), 8)],
        ['RTE/QNLI λ=1.5', '2 tasks × 3 seeds', f'{lambda15}/6', status(lambda15, 6)],
        ['QNLI λ=0.7', '1 task × 3 seeds', f'{lambda07}/3', status(lambda07, 3)],
        ['Rank permutation', 'MRPC/RTE × 3 seeds', f"{complete(D['rank_shuffle'])}/6", status(complete(D['rank_shuffle']), 6)],
    ]
    add('11_completion_status', '实验完成情况', ['Experiment family','Planned result cells','Complete','Status'], rows,
        f"Availability snapshot: {D['snapshot_finished'].replace('T',' ')}. Counts are result cells, including reused "
        'completed runs, not independent new training jobs; rows overlap and must not be summed. Smoke tests, '
        'failed runs and incomplete epochs are excluded. Main-ablation v1/v2 and R-Drop v1 failed during preflight '
        'and are not result-bearing versions. Planned mechanism or budget studies without outputs are not presented as findings.',
        align='llrl')
    rows = []
    for task in ['cola','mrpc','rte']:
        for v in 'ABCDEF':
            scores = [value(component(task,v,s)) for s in SEEDS]
            actual = [x for x in scores if x is not None]
            rows.append([LABELS[task],v] + [num(x) for x in scores] + [str(len(actual)),avg(actual)])
    add('12_component_seed_detail', '主消融逐种子明细',
        ['Task','Variant','Seed 42','Seed 777','Seed 2026','n','Mean ± SD'], rows,
        'All 54 planned component-ablation cells, with only completed final metrics filled. A–F follow Table 2. '
        'n counts completed seeds. Means with n<3 are partial summaries and must not be described as full three-seed '
        'results; n=1 has no SD. Reused results are counted once per planned cell. This appendix supports auditing '
        'without mixing single-seed ablations with the completed main-table means.', align='llrrrrr', breaks=[6,12])
    rows = []
    total = 0
    for task in TASKS:
        hours = [r['elapsed_seconds']/3600 for r in main_runs(task)]
        total += sum(hours)
        rows.append([LABELS[task],str(main_selected(task)['epochs']),num(stats.mean(hours)),
                     f'{min(hours):.2f}–{max(hours):.2f}',num(sum(hours))])
    rows.append(['Total','','','','**'+num(total)+'**'])
    add('13_recorded_runtime', '历史运行耗时', ['Task','Epochs','Mean/run (h)','Range (h)','Three seeds (h)'], rows,
        'Recorded single-GPU elapsed time for the selected 24 GLUE runs. The timer covers training, periodic '
        'validation and checkpoint saving, but excludes model loading and calibration/initialization. Values reflect '
        'historical hardware/load conditions and are not a controlled efficiency benchmark. The final column sums '
        'run times; it is not the wall time of a parallel campaign.', align='lrrrr', breaks=[8])


def completed_extensions():
    rows = []
    for method, label, params in [('uniform', 'Uniform LoRA r=8', 1327104), ('speccon', 'SpecCon-LoRA', 755712)]:
        for alpha in (0, 1):
            cells = []
            for task in ('cola', 'mrpc', 'rte'):
                group = [x['result'] for x in D['extended_rdrop'] if x['task'] == task and
                         x['method'] == method and x['alpha'] == alpha and x['result']]
                assert sorted(x['seed'] for x in group) == SEEDS
                assert all(x['parameters'] == params for x in group)
                cells.append(avg([x['score'] for x in group]))
            rows.append([label, f'{params/1e6:.3f}', str(alpha), '3'] + cells)
    add('14_rdrop_multiseed', 'R-Drop 消融（三种子）',
        ['Method', 'Params (M)', 'α', 'n/task', 'CoLA', 'MRPC', 'RTE'], rows,
        'Completed R-Drop extension: all 36 cells over seeds 42/777/2026. Values are final-epoch mean ± sample SD. '
        'SpecCon λ is 1.5/1.5/0.5 on CoLA/MRPC/RTE. Within-method budgets and task settings are fixed; '
        'cross-method budgets differ. α=0/1 uses one/two forwards, respectively. MRPC retains label smoothing=0.05. '
        'These are recipe comparisons, not equal-compute KL-only ablations.', align='lrrrrrr', breaks=[2])
    rows = []
    for task in ('mrpc', 'rte'):
        original = {x['seed']: x for x in D['selected'][task]}
        shuffled = {x['seed']: x['result'] for x in D['rank_shuffle'] if x['task'] == task and x['result']}
        assert sorted(original) == sorted(shuffled) == SEEDS
        for seed in SEEDS:
            assert original[seed]['parameters'] == shuffled[seed]['parameters'] == 755712
            assert original[seed]['hyperparameters']['lambda'] == shuffled[seed]['hyperparameters']['lambda']
        for label, group in [('Original', original), ('Permuted', shuffled)]:
            scores = [group[s]['score'] for s in SEEDS]
            rows.append([LABELS[task], label] + [num(s) for s in scores] + [avg(scores)])
    add('15_rank_permutation', '秩放置置换对照（三种子）',
        ['Task', 'Rank placement', 'Seed 42', 'Seed 777', 'Seed 2026', 'Mean ± SD'], rows,
        'Rank-placement control, fixed final accuracy (×100). All runs retain 755,712 parameters and the rank '
        'multiset within each matrix-shape group; ranks are permuted only among same-shape modules. '
        'Permutation seeds 101/202/303 are paired with training seeds 42/777/2026 without selecting among '
        'permutations. Each module rebuilds its own gradient-SVD initialization after rank placement; B remains zero. '
        'MRPC/RTE use λ=1.5/0.5 and R-Drop α=1. Original runs are reused.', align='llrrrr', breaks=[2])


def escape(text):
    translation = {'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
                   '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
                   '±':r'$\pm$','λ':r'$\lambda$','α':r'$\alpha$','γ':r'$\gamma$',
                   'Δ':r'$\Delta$','×':r'$\times$','−':r'$-$','—':r'---','–':r'--',
                   '≤':r'$\leq$'}
    pieces = str(text).split('**')
    return ''.join((r'\textbf{' if i%2 else '') + ''.join(translation.get(c,c) for c in p) +
                   ('}' if i%2 else '') for i,p in enumerate(pieces))


def export():
    for subdir in ['tables','csv']:
        (OUT/subdir).mkdir(exist_ok=True)
    md = [f'# 实验与消融汇总（ACL 表格）', '', f"数据快照：{D['snapshot_finished']}。", '',
          '主结果为固定末轮验证集成绩；多种子使用样本标准差（ddof=1）。— 表示没有完整结果，不代表零。', '',
          'GLUE 主表 QNLI 更新为非零 λ=0.7，其余沿用此前配置；λ=0 保留为消融对照。数学、代码为保留的最高单次记录。', '',
          'ACL 排版依据：[官方模板](https://github.com/acl-org/acl-style-files)及[格式说明](https://acl-org.github.io/ACLPUB/formatting.html)。', '',
          'DPLoRA 论文对照：[Park et al. (2026), Table 8](https://aclanthology.org/2026.findings-acl.1693.pdf)。', '']
    wb=Workbook();wb.remove(wb.active)
    thin=Side(style='thin',color='000000');thick=Side(style='medium',color='000000')
    for index,t in enumerate(TABLES,1):
        caption=escape(t['caption'])
        if t['id']=='01_glue_main':
            caption=caption.replace('Park et al. (2026)',r'\citet{park-etal-2026-dplora}')
        lines=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',
               r'\renewcommand{\arraystretch}{1.12}',
               r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+t['align']+'@{}}',
               r'\toprule',' & '.join(escape(x) for x in t['headers'])+r' \\',r'\midrule']
        for i,row in enumerate(t['rows']):
            if i in t['breaks']:lines.append(r'\midrule')
            lines.append(' & '.join(escape(x) for x in row)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular*}',r'\caption{'+caption+'}',r'\label{tab:'+t['id']+'}',r'\end{table*}','']
        (OUT/'tables'/f"{t['id']}.tex").write_text('\n'.join(lines))
        with (OUT/'csv'/f"{t['id']}.csv").open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(t['headers']);writer.writerows([[x.replace('**','') for x in row] for row in t['rows']])
        md += [f"## 表 {index}：{t['title']}",'','| '+' | '.join(t['headers'])+' |',
               '| '+' | '.join('---' for _ in t['headers'])+' |']
        md += ['| '+' | '.join(row)+' |' for row in t['rows']]
        md += ['',t['caption'],'']
        ws=wb.create_sheet(t['id'][:31]);ws.append(t['headers'])
        for row in t['rows']:ws.append([x.replace('**','') for x in row])
        for row in ws:
            for cell in row:
                cell.font=Font(name='Times New Roman',size=11,bold=cell.row==1)
                cell.alignment=Alignment(vertical='center',horizontal='left' if cell.column==1 else 'center')
                if cell.row==1:cell.border=Border(top=thick,bottom=thin)
                if cell.row==len(t['rows'])+1:cell.border=Border(bottom=thick)
        for c in range(1,len(t['headers'])+1):
            width=max(len(str(ws.cell(r,c).value or '')) for r in range(1,ws.max_row+1))+3
            ws.column_dimensions[get_column_letter(c)].width=min(45,max(12,width))
        ws.freeze_panes='B2';ws.sheet_view.showGridLines=False
        note_row=ws.max_row+3;ws.cell(note_row,1,t['caption'])
        ws.merge_cells(start_row=note_row,start_column=1,end_row=note_row+2,end_column=len(t['headers']))
        ws.cell(note_row,1).alignment=Alignment(wrap_text=True,vertical='top')
        ws.cell(note_row,1).font=Font(name='Times New Roman',size=10)
    ws=wb.create_sheet('sources');ws.append(['Original source','SHA-256','Frozen copy'])
    for path,metadata in D['source_files'].items():ws.append([path,metadata['sha256'],metadata['snapshot']])
    ws.freeze_panes='A2';ws.column_dimensions['A'].width=90;ws.column_dimensions['B'].width=68;ws.column_dimensions['C'].width=70
    wb.save(OUT/'experiment_tables.xlsx')
    (OUT/'tables_zh.md').write_text('\n'.join(md)+'\n')
    (OUT/'table_data.json').write_text(json.dumps(TABLES,ensure_ascii=False,indent=2)+'\n')
    # Each group occupies one or more full-width float pages in the official two-column class.
    groups=[[0,1],[2,3,4],[5,6,7],[8,9],[10,11],[12],[13,14]]
    includes=[]
    for group in groups:
        includes += [r'\input{tables/'+TABLES[i]['id']+'.tex}' for i in group]
        includes += [r'\clearpage']
    (OUT/'all_tables.tex').write_text('\n'.join(includes)+'\n')
    (OUT/'acl_tables.tex').write_text(r'''\documentclass[11pt]{article}
\usepackage[final]{acl}
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{array}
\begin{document}
\input{all_tables}
\bibliography{references}
\end{document}
''')
    (OUT/'references.bib').write_text(r'''@inproceedings{park-etal-2026-dplora,
  title = {{DPLoRA}: A Dual-Pruning Framework based on {ILP} Optimization and Progressive Pruning for Parameter-Efficient {LoRA} Fine-Tuning},
  author = {Park, Changjun and Yoon, Sejong and Kim, Jaekwang},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year = {2026},
  pages = {33892--33908},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2026.findings-acl.1693},
  url = {https://aclanthology.org/2026.findings-acl.1693/}
}
''')
    (OUT/'README.md').write_text(f'''# ACL 实验表格包

数据快照：{D['snapshot_finished']}。共 {len(TABLES)} 张三线表。

- `acl_tables.pdf`：使用 ACL 官方样式实际编译的表格合集。
- `acl_tables.tex`：完整 LaTeX 编译入口；`tables/`：可单独粘贴或 `\\input` 到论文的各表。
- `experiment_tables.xlsx`：便于修改的 Excel 表格，末页包含全部输入来源。
- `tables_zh.md`：中文表名与完整数据；`csv/`：每张表的 CSV。
- `snapshot.json`：未舍入的指标、配置、运行状态及源文件哈希；`sources/`：只读输入的冻结副本。
- `collect_results.py`：重新采集时会更新本目录快照；`build_tables.py` 只使用已冻结快照生成表格。

默认英文表头与英文 caption，采用 `booktabs` 三线表、无竖线、双栏跨栏 `table*`。
模板来自 https://github.com/acl-org/acl-style-files ，未修改 `acl.sty`。
版式按 https://acl-org.github.io/ACLPUB/formatting.html ：A4、Times、双栏、10pt 表注、表注放在表格下方。
这是实验表格合集，不是已写完的论文。正式论文中将需要的 `tables/*.tex` 放到相关段落附近即可。

统计与范围：

1. GLUE 使用固定末轮验证结果，三种子为 42/777/2026，样本标准差 ddof=1；MRPC 统一为 Accuracy。
2. 主表 QNLI 更新为 λ=0.7，三个种子全部完成；其余任务沿用此前配置。原 λ=0 结果保留在消融表。DPLoRA 论文结果与本地复现分表呈现。
3. A/B 的 uniform r8 有 1,327,104 个参数；C–F 有 755,712 个参数。跨这两个预算组不能声称是纯分配消融。
4. R-Drop 关闭时为一次前向、开启时两次；这不是等计算量的纯 KL 消融。
5. 未完成单元格用 —，部分种子的均值标明 n，不填造假标准差；预检与失败版本均排除。
6. 数学和代码是清理后保留的最高单次结果；MT-Bench 是历史单种子评分，训练配方和裁判日期不同。
7. 不重建已删除且没有完整指标证据的旧调参试验；保留结果及当前消融构成此表格包的范围。
8. 正在运行的实验不会被修改、停止或重启；本文件只代表上述时间快照。

本地重建：`python build_tables.py`，然后 `tectonic --keep-logs acl_tables.tex`；也可上传源码至 Overleaf 使用 pdfLaTeX。
源码、数据和图片均位于本目录，便携式编译器单独存于 `/workspace/.tools/tectonic-0.17.0/tectonic`。
''')
    print(json.dumps({'tables':len(TABLES),'snapshot':D['snapshot_finished'],'output':str(OUT)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main_results();components();rdrop();lambda_tables();historical();appendices();completed_extensions();export()
