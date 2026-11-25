#!/usr/bin/env python3
"""Maintain a truthful chapter-4 report as long GPU-1 queues finish."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECT = Path('/workspace/SpecCon-LoRA')
BASE = PROJECT / 'experiments'
ONE = BASE / 'mrpc_calibration_robustness_v1'
TWO = BASE / 'rank_utility_predictivity_v1'
EXTENSION = TWO / 'seed_extension'
THREE = BASE / 'geometric_predictivity_validation_v1'
FINAL = BASE / 'CHAPTER4_MECHANISM_VALIDATION_REPORT.md'
CONTROLLERS = BASE / 'chapter4_mechanism_controller_pids.json'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def refresh_analysis():
    for script in ('analyze_mrpc_calibration_robustness_v1.py',
                   'analyze_rank_utility_predictivity_v1.py'):
        subprocess.check_call([sys.executable, str(PROJECT / 'reproduction' / script)])


def check_controllers():
    pids=read(CONTROLLERS,{})
    for name,folder,expected,terminal in (
        ('exp01',ONE,'run_mrpc_calibration_robustness_v1.py',('complete','failed','stopped')),
        ('exp02',TWO,'run_rank_utility_predictivity_v1.py',('complete','failed','stopped')),
        ('conditional_seed_extension',EXTENSION,'run_rank_utility_seed_extension_v1.py',
         ('complete','no_predeclared_signal','failed','stopped','upstream_failed'))):
        state=read(folder/'status.json',{})
        if not state or state.get('state') in terminal:
            continue
        pid=pids.get(name)
        cmdline=Path(f'/proc/{pid}/cmdline') if pid else None
        if cmdline is None or not cmdline.exists() or expected.encode() not in cmdline.read_bytes():
            state.update(state='failed',failure=f'{name} controller PID {pid} exited; inspect controller.log',
                         failed_detection='report_watcher')
            path=folder/'status.json';tmp=path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def build():
    one_status = read(ONE / 'status.json', {})
    two_status = read(TWO / 'status.json', {})
    one_done = read(ONE / 'completed_new.json', {})
    two_done = read(TWO / 'completed.json', {})
    extension_status = read(EXTENSION / 'status.json', {})
    extension_analysis = read(EXTENSION / 'analysis.json', {})
    three = (THREE / 'REPORT.md').read_text() if (THREE / 'REPORT.md').exists() else ''
    analysis_two = read(TWO / 'analysis.json', {})
    import csv
    with (ONE / 'paired_results.csv').open(newline='') as stream:
        paired = list(csv.DictReader(stream))
    with (ONE / 'runs.csv').open(newline='') as stream:
        runs = list(csv.DictReader(stream))
    d = [float(row['delta_consensus_minus_energy']) for row in paired]
    seed42 = [float(row['delta_consensus_minus_energy']) for row in paired if int(row['training_seed'])==42]
    wins=sum(v>0 for v in d); ties=sum(v==0 for v in d)
    one_complete = len(runs)==30 and len(paired)==15 and one_status.get('state')=='complete'
    two_complete = bool(analysis_two.get('complete')) and two_status.get('state')=='complete'
    lines = ['# SpecCon-LoRA 第四章核心机制验证总报告', '',
        '**状态：**' + ('三项实验均完成。' if one_complete and two_complete and bool(three)
                      and extension_status.get('state') in ('complete','no_predeclared_signal')
                      else '实时报告；EXP-01/02 长时训练尚未全部完成，未填入不存在的结果。'), '',
        '## 1. 已完成实验与本次新增实验统计', '',
        '|实验|历史可复用|本次计划新增|当前已完成|状态|',
        '|---|---:|---:|---:|---|',
        f'|EXP-01 MRPC 跨校准|10|20|{len(one_done)}|{one_status.get("state","未知")}|',
        f'|EXP-02 rank swap 参考＋候选|0|13|{len(two_done)}|{two_status.get("state","未知")}|',
        f'|EXP-02 条件性额外种子|0|若触发则26|{extension_status.get("complete",0)}|{extension_status.get("state","尚未开始")}|',
        '|EXP-03 几何统计|冻结 A/B 校准数据|0 次训练|统计已完成|完成|', '',
        'EXP-01 使用五个原始 A 子集、512 样本、四块、相同 seed42 模型及分类头；'
        'seed777/2026 只改变训练数据顺序和恢复后的训练 RNG。'
        'A子集与历史默认校准（训练集前512条即时采集）的绝对分数对比还包含校准阶段'
        'dropout随机流的差异；A组内部lambda配对仍共享相同缓存梯度。'
        'EXP-02 的 12 个候选在训练前由 2,520 个预算守恒候选预注册，'
        '并因历史 near-uniform 校准 ID 不同而增加一项 A0 匹配参考。', '',
        '## 2. EXP-01 跨校准集可靠性', '',
        f'当前完成 {len(runs)}/30 项、{len(paired)}/15 对。'
        f'已观测配对 Δ 的描述性均值 {statistics.mean(d):+.3f} pp，'
        f'样本 SD {statistics.stdev(d):.3f} pp；胜/平/负 {wins}/{ties}/{len(d)-wins-ties}。',
        f'原 seed42 五对均值 {statistics.mean(seed42):+.3f} pp。',
        '完整逐子集、逐种子结果和 rank 图见 [EXP-01 报告](mrpc_calibration_robustness_v1/REPORT.md)。'
        'A 子集可重叠，共用固定模型及分类头；15 对不是独立观测。', '',
        '## 3. EXP-02 真实 rank 效用预测', '',
        f'预注册 12 个同形状、等成本的一增一减交换；当前完成 {len(two_done)}/13 项。'
        '参考和候选的 A0 校准、训练 seed、R-Drop、优化器、预算及最终 checkpoint 一致。',
        '候选、三种评分与全部可行交换清单见 [预注册文件](rank_utility_predictivity_v1/pre_registered_swaps.json)；'
        '逐项观察值见 [EXP-02 报告](rank_utility_predictivity_v1/REPORT.md)。']
    if analysis_two.get('complete'):
        s={row['method']:row for row in analysis_two['statistics']}
        lines += ['', '|评分|Spearman ρ|方向命中率|Top-3 实际平均 Δ|',
                  '|---|---:|---:|---:|']
        for method in ('raw_spectral_energy','energy_only_utility','consensus_adjusted_utility'):
            row=s[method]
            lines.append(f'|{method}|{row["spearman_rho"]:+.3f}|'
                f'{row["direction_accuracy"]:.1%}|{row["top3_actual_mean_delta"]:+.3f} pp|')
    else:
        lines.append('排序相关、方向命中率及 Top-3 真实收益将在12个预注册交换全部完成后给出。')
    if extension_status.get('state')=='no_predeclared_signal':
        lines.append('预先冻结的扩展判据未满足，未启动额外种子训练；判据和实测值见 seed_extension/decision.json。')
    elif extension_analysis.get('complete_seeds'):
        lines += ['', '固定12个交换的额外训练种子复核：', '',
                  '|训练seed|Raw energy ρ|Energy-only ρ|Consensus ρ|',
                  '|---:|---:|---:|---:|']
        for row in extension_analysis['complete_seeds']:
            lines.append(f'|{row["training_seed"]}|{row["raw_spectral_energy_spearman"]:+.3f}|'
                f'{row["energy_only_utility_spearman"]:+.3f}|'
                f'{row["consensus_adjusted_utility_spearman"]:+.3f}|')
    lines += ['', '## 4. EXP-03 几何相关性统计验证', '',
        '冻结数据经独立统计复核后，在控制 rank、模块类型和谱能量下，'
        'Consensus 与 held-out recurrence 的相关分别为 CoLA 0.953、MRPC 0.949、RTE 0.932；'
        '999 次模块曲线分层置换的双侧 p 均为 0.001。'
        'Raw energy 的相应未条件相关分别为 0.204、0.148、0.284。'
        '模块分组 Bootstrap 及全部控制统计见 [EXP-03 报告](geometric_predictivity_validation_v1/REPORT.md)。'
        '在每个模块形状和 rank 下，还将实测 A/B 重现度与独立各向同性子空间的解析期望 k/d 对比；'
        '252 个分层格的实测均值均高于该维度基线。'
        '置换保留模块类型、nested rank 曲线及粗能量分层；区间不覆盖共用 B 和 A 重叠。', '',
        '## 5. 对核心创新的证据评估', '',
        '目前几何预测证据为正：共识分数包含超出原始谱能量的 held-out 几何信息。'
        '但原 seed42 的五组独立校准下游平均 Δ 为负。'
        '几何信息能否指导真实 rank 增减，须等 EXP-02 完整训练；'
        '跨训练随机性的可靠性须等 EXP-01 另两种子。', '',
        '## 6. 可以写入第四章的结果', '',
        '可以写入 EXP-03 的受控几何相关、模块置换方法和不确定性限制；'
        '可以如实写入原 seed42 五组跨校准负向结果。'
        'EXP-01/02 的新增结果只有在对应矩阵完整后才作完整结论。', '',
        '## 7. 不应直接宣称的结论', '',
        '目前不应宣称 Consensus 普遍提高跨校准稳定性或下游 Accuracy，'
        '也不能把高几何相关解释为 rank 交换的真实收益预测已经验证。'
        '模块级 Bootstrap/置换不等于独立任务或独立 held-out B 的重复。', '',
        '## 8. 是否需要调整第 3.2 节 Consensus 评分公式', '',
        '现阶段不修改冻结公式。EXP-02 全部结果若显示共识效用未优于 Energy-only，'
        '应先收窄公式的经验主张，再考虑新的独立诊断；不能按这批结果反向调参。', '',
        '## 9. 下一步建议', '',
        '持续核验 GPU 1 的 EXP-01 队列及首对一致性报告；完成20项后，'
        'EXP-02 队列自动开始 A0 参考和12个预注册交换。'
        '若第一阶段有预先定义的可解释预测信号，再对同一批候选做额外种子复核。', '',
        '所有原始训练日志、配置、样本 ID、rank 图和最终指标保存在各实验目录；'
        '历史实验不作覆盖。']
    FINAL.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    previous=None
    while True:
        check_controllers()
        stamp=tuple((p.stat().st_mtime_ns if p.exists() else None) for p in
            [ONE/'completed_new.json',TWO/'completed.json',ONE/'status.json',TWO/'status.json',
             EXTENSION/'status.json',EXTENSION/'analysis.json'])
        if stamp!=previous:
            refresh_analysis();build();previous=stamp
        if not args.watch:
            break
        a=read(ONE/'status.json',{});b=read(TWO/'status.json',{})
        if a.get('state') in ('failed','stopped') or b.get('state') in ('failed','stopped'):
            break
        extension=read(EXTENSION/'status.json',{})
        if a.get('state')=='complete' and b.get('state')=='complete' and \
                extension.get('state') in ('complete','no_predeclared_signal'):
            break
        time.sleep(60)


if __name__=='__main__':
    main()
