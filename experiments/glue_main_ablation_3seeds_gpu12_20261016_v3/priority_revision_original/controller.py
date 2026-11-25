#!/usr/bin/env python3
"""Run the six-cell GLUE allocation/init matrix concurrently on GPUs 1 and 2."""
import argparse
import ast
import copy
import csv
import difflib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime

PROJECT = Path('/workspace/SpecCon-LoRA')
ROOT = PROJECT / 'experiments/glue_main_ablation_3seeds_gpu12_20261016_v3'
BASE_MANIFEST = PROJECT / 'reproduction/current_glue_manifest.json'
SUPPORT = PROJECT / 'reproduction/main_ablation_support.py'
RDROP = PROJECT / 'experiments/glue_rdrop_ablation_seed42_gpu3_20261016_v2'
ENERGY = PROJECT / 'experiments/glue_lambda0_seed42_gpu2_20261015_v1'
TASKS = ['cola', 'mrpc', 'rte']
SEEDS = [42, 777, 2026]
GPUS = [1, 2]
MIN_FREE_MIB = {'cola': 6144, 'mrpc': 9216, 'rte': 14336}


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one occurrence of {old!r}, found {source.count(old)}')
    return source.replace(old, new)


def set_option(command, option, value):
    if option in command:
        command[command.index(option)+1] = str(value)
    else:
        command.extend([option, str(value)])


def budget(variant):
    return 1327104 if variant in 'AB' else 755712


def expected_hp(row, variant, rng):
    hp = copy.deepcopy(row['hyperparameters'])
    hp.update(ablation_variant=variant, adapter_parameter_budget=budget(variant),
        training_rng_reference=str(rng), rank_range=[8, 8] if variant in 'AB' else [2, 16],
        init_method='svd_a_zero_b' if variant in 'BDF' else 'none',
        calibration_steps=0 if variant == 'A' else 16,
        calibration_blocks=0 if variant == 'A' else 4,
        gamma=None if variant in 'AB' else 1.2)
    hp['lambda'] = None if variant in 'AB' else (0.0 if variant in 'CD' else row['hyperparameters']['lambda'])
    assert hp['rdrop_alpha'] == 1.0 and hp['training_forwards_per_step'] == 2
    return hp


def canonical_hp(hp):
    ignored = {'training_rng_reference', 'adapter_method', 'adapter_parameter_budget', 'ablation_variant'}
    return {key: value for key, value in hp.items() if key not in ignored}


def primary_score(metric):
    key = 'matthews_correlation' if metric['task'] == 'cola' else 'accuracy'
    result = 100 * metric['final']['splits']['validation'][key]
    assert math.isfinite(result)
    return result


def patch_runner(original):
    source = replace_once(original,
        'from speccon_lora import SpecConLoraConfig, prepare_specconlora_model',
        'from speccon_lora import SpecConLoraConfig, prepare_specconlora_model\n'
        'from main_ablation_support import prepare_main_ablation')
    extra_args = ('    parser.add_argument("--ablation-variant", choices=list("ABCDEF"), required=True)\n'
                  '    parser.add_argument("--allocation-reference", type=Path, required=True)\n'
                  '    parser.add_argument("--expected-allocation", type=Path)\n')
    if 'parser.add_argument("--subspace-consistency-lambda"' not in source:
        extra_args += '    parser.add_argument("--subspace-consistency-lambda", type=float, default=1.5)\n'
        source = replace_once(source, 'subspace_consistency_lambda=1.5,',
            'subspace_consistency_lambda=args.subspace_consistency_lambda,')
    source = replace_once(source, '    args = parser.parse_args()\n', extra_args + '    args = parser.parse_args()\n')
    source = replace_once(source, 'def main():\n', 'def main():\n    global PARAM_BUDGET\n')
    marker = '    args = parse_args(); cfg = dict(TASK_CONFIG[args.task]); output = Path(args.output_dir)\n'
    source = replace_once(source, marker, marker +
        '    PARAM_BUDGET = 1_327_104 if args.ablation_variant in "AB" else 755_712\n')
    source = replace_once(source, '    cfg["calibration_steps"] = calibration_steps\n',
        '    cfg["ablation_variant"] = args.ablation_variant\n'
        '    cfg["adapter_parameter_budget"] = PARAM_BUDGET\n'
        '    cfg["calibration_steps"] = 0 if args.ablation_variant == "A" else calibration_steps\n')
    source = replace_once(source, '    cfg["calibration_blocks"] = 4\n',
        '    cfg["calibration_blocks"] = 0 if args.ablation_variant == "A" else 4\n')
    source = replace_once(source,
        '    model, report = prepare_specconlora_model(model, calibration_loader, loss_fn, speccon, device=device)',
        '    model, report = prepare_main_ablation(model, calibration_loader, loss_fn, speccon,\n'
        '        device, args.ablation_variant, args.allocation_reference, args.expected_allocation, output)')
    source = replace_once(source, '"model": "RoBERTa-base", "method": "SpecCon-LoRA",',
        '"model": "RoBERTa-base", "method": "MainAblation-" + args.ablation_variant,')
    source = replace_once(source, '"rank_range": [2, 16], "init_method": "svd_a_zero_b",',
        '"rank_range": [speccon.r_min, speccon.r_max], "init_method": speccon.init_method,')
    old = ('"gamma": 1.2, "lambda": 1.5,' if '"gamma": 1.2, "lambda": 1.5,' in source
           else '"gamma": args.rank_score_gamma, "lambda": args.subspace_consistency_lambda,')
    source = replace_once(source, old,
        '"gamma": None if args.ablation_variant in "AB" else speccon.rank_score_gamma,\n'
        '                            "lambda": None if args.ablation_variant in "AB" else speccon.subspace_consistency_lambda,')
    ast.parse(source)
    start = '    epochs = args.smoke_epochs if args.smoke else cfg["epochs"]'
    finish = '    result = {\n'
    assert original.split(start)[1].split(finish)[0] == source.split(start)[1].split(finish)[0]
    original_defs = {node.name: ast.dump(node) for node in ast.parse(original).body if isinstance(node, ast.FunctionDef)}
    changed_defs = {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    assert all(original_defs[name] == changed_defs[name] for name in original_defs if name not in {'main', 'parse_args'})
    return source


def reuse_source(row, variant):
    if variant == 'F':
        return Path(row['source'])
    if row['seed'] == 42 and variant == 'D':
        return ENERGY / 'runs' / row['task'] / 'seed42'
    if row['seed'] == 42 and variant == 'A':
        return RDROP / 'runs' / row['task'] / 'uniform_rd1/seed42'
    return None


def validate_reuse(row, variant, source):
    metric = read(source / 'metrics.json')
    assert metric['state'] == 'complete' and metric['selected'] == metric['final']
    assert metric['task'] == row['task'] and metric['seed'] == row['seed']
    assert metric['epochs'] == row['hyperparameters']['epochs']
    assert metric['adapter_parameters'] == budget(variant)
    assert metric['precision'] == 'FP32' and metric['global_batch_size'] == 32
    target_hp = expected_hp(row, variant, '')
    assert canonical_hp(metric['hyperparameters']) == canonical_hp(target_hp), (source, metric['hyperparameters'], target_hp)
    expected_rng = read(Path(row['source']) / 'training_rng.json')
    actual_rng = read(source / 'training_rng.json')
    assert all(actual_rng[key] == expected_rng[key] for key in ['cpu_sha256', 'cuda_sha256'])
    assert (source / 'final_adapter/adapter_model.safetensors').is_file()
    allocation = read(source / 'allocation.json')
    assert allocation['rank_report']['budget_used'] == budget(variant)
    assert len(allocation['rank_pattern']) == 72
    if variant == 'A':
        assert set(allocation['rank_pattern'].values()) == {8}
    return metric


def prepare():
    records = {(row['task'], row['seed']): row for row in read(BASE_MANIFEST)['runs']
               if row['task'] in TASKS and row['seed'] in SEEDS}
    assert len(records) == 9
    for row in records.values():
        assert digest(row['runner']) == row['runner_sha256']
        patch_runner(Path(row['runner']).read_text())
        for variant in 'ADF':
            source = reuse_source(row, variant)
            if source:
                validate_reuse(row, variant, source)
    ast.parse(SUPPORT.read_text())
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'logs').mkdir()
    (ROOT / 'preflight').mkdir()
    shutil.copy2(BASE_MANIFEST, ROOT / 'original_manifest.json')
    cells, jobs, references = [], [], []
    for task in TASKS:
        for seed in SEEDS:
            row = records[task, seed]
            source = Path(row['source'])
            snapshot = ROOT / 'source_snapshot' / task / f'seed{seed}'
            shutil.copytree(Path(row['cwd']) / 'speccon_lora', snapshot / 'speccon_lora',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            (snapshot / 'reproduction').mkdir()
            original = Path(row['runner']).read_text()
            runner = snapshot / 'reproduction' / Path(row['runner']).name
            modified = patch_runner(original)
            runner.write_text(modified)
            shutil.copy2(SUPPORT, snapshot / 'main_ablation_support.py')
            reference = ROOT / 'references' / task / f'seed{seed}'
            reference.mkdir(parents=True)
            for filename in ['metrics.json', 'run_config.json', 'allocation.json', 'training_rng.pt', 'training_rng.json']:
                shutil.copy2(source / filename, reference / filename)
            shutil.copy2(row['runner'], reference / 'original_runner.py')
            (reference / 'runner.patch').write_text(''.join(difflib.unified_diff(
                original.splitlines(True), modified.splitlines(True), fromfile=row['runner'], tofile=str(runner))))
            for variant in 'ABCDEF':
                cell_id = f'{task}_{variant}_s{seed}'
                output = ROOT / 'runs' / task / variant / f'seed{seed}'
                command = list(row['command'])
                command[0], command[2] = sys.executable, str(runner)
                lam = row['hyperparameters']['lambda'] if variant in 'EF' else 0.0
                for option, value in [('--output-dir', output), ('--ablation-variant', variant),
                        ('--rdrop-alpha', 1.0), ('--subspace-consistency-lambda', lam),
                        ('--training-rng-reference', reference / 'training_rng.pt'),
                        ('--allocation-reference', reference / 'allocation.json')]:
                    set_option(command, option, value)
                allocation_reference = None
                if variant in 'EF':
                    allocation_reference = reference / 'allocation.json'
                elif variant in 'CD' and seed == 42:
                    allocation_reference = reference / 'D_allocation.json'
                    if not allocation_reference.exists():
                        shutil.copy2(ENERGY / 'runs' / task / 'seed42/allocation.json', allocation_reference)
                if allocation_reference:
                    set_option(command, '--expected-allocation', allocation_reference)
                job = dict(id=cell_id, task=task, seed=seed, variant=variant,
                    adapter_parameters=budget(variant), command=command, cwd=str(snapshot),
                    output=str(output), log=str(ROOT / 'logs' / (cell_id + '.log')),
                    expected_hyperparameters=expected_hp(row, variant, reference / 'training_rng.pt'),
                    runner_sha256=digest(runner), reference=str(reference),
                    min_free_mib=MIN_FREE_MIB[task])
                reused = reuse_source(row, variant)
                if reused:
                    metric = validate_reuse(row, variant, reused)
                    destination = reference / f'reused_{variant}'
                    destination.mkdir()
                    for filename in ['metrics.json', 'run_config.json', 'allocation.json', 'training_rng.json']:
                        shutil.copy2(reused / filename, destination / filename)
                    initial = reused / 'initial_state.json'
                    if initial.exists():
                        shutil.copy2(initial, destination / initial.name)
                    references.append(dict(id=cell_id, task=task, seed=seed, variant=variant,
                        adapter_parameters=budget(variant), reused=True, source=str(reused),
                        output=str(reused), metrics=str(destination / 'metrics.json'),
                        allocation=str(destination / 'allocation.json'), score=primary_score(metric)))
                    job['reused'] = True
                else:
                    job['reused'] = False
                    job['gpu'] = GPUS[len(jobs) % len(GPUS)]
                    jobs.append(job)
                cells.append(job)
    assert len(cells) == 54 and len(jobs) == 39 and len(references) == 15
    preflight = []
    # Six one-epoch checks cover every mode, all three frozen runners, and paired rank maps.
    for task, variant in [('cola', 'A'), ('cola', 'B'), ('mrpc', 'C'), ('mrpc', 'D'), ('rte', 'E'), ('rte', 'F')]:
        job = copy.deepcopy(next(cell for cell in cells if (cell['task'], cell['seed'], cell['variant']) == (task, 42, variant)))
        job['id'] = 'smoke_' + job['id']
        job['smoke'], job['reused'] = True, False
        job.pop('gpu', None)
        job['output'] = str(ROOT / 'preflight' / job['id'])
        job['log'] = str(ROOT / 'logs' / (job['id'] + '.log'))
        set_option(job['command'], '--output-dir', job['output'])
        job['command'].append('--smoke')
        preflight.append(job)
    shutil.copy2(__file__, ROOT / 'controller.py')
    paths = list((ROOT / 'source_snapshot').rglob('*.py')) + [ROOT / 'controller.py']
    paths += [p for p in (ROOT / 'references').rglob('*') if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    manifest = dict(created=now(), gpus=GPUS, seeds=SEEDS, tasks=TASKS, cells=cells,
        jobs=jobs, references=references, preflight=preflight, source_sha256=hashes,
        total_cells=54, reused_cells=15, new_runs=39,
        budget_note='User-retained budgets: A/B uniform rank8=1327104; C-F adaptive=755712. D-B and the cross-family interaction are not budget-controlled.',
        scheduling='User explicitly requested concurrent addition on GPUs 1/2 while existing experiments continue. One new process per GPU; task-specific free-memory checks. No dependency on older queues.',
        allocator='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')
    write(ROOT / 'manifest.json', manifest)
    write(ROOT / 'status.json', dict(state='prepared', updated=now()))
    (ROOT / 'README.md').write_text('''# 主消融：3 种分配 × 2 种初始化

CoLA、MRPC、RTE × A–F × seed 42/777/2026，共 54 个结果单元。
复用 F 的 9 次、D 的 3 次 seed42，以及上一组 R-Drop 实验 A 的 3 次 seed42，新增 39 次。

|组|分配|初始化|参数量|
|---|---|---|---:|
|A|uniform r8|标准随机 A / 零 B|1,327,104|
|B|uniform r8|梯度 SVD A / 零 B|1,327,104|
|C|分块谱，λ=0|标准随机 A / 零 B|755,712|
|D|分块谱，λ=0|梯度 SVD A / 零 B|755,712|
|E|分块谱＋一致性|标准随机 A / 零 B|755,712|
|F|分块谱＋一致性|梯度 SVD A / 零 B|755,712|

CoLA/MRPC/RTE 分别训练与调度 42/57/63 轮，E/F 的 λ 分别为 1.5/1.5/0.5。
全部 R-Drop α=1、FP32、batch32、缩放2、A/B 相同学习率；MRPC label smoothing=0.05。
F 的冻结配置与训练代码来自 current_glue_manifest.json，原始优化/训练/评测循环逐字核验保持不变。
校准仍为训练集前16批、4个块；C/D 保留同样的分块谱，只把一致性权重设为0。
A 不做校准；B 固定rank8，并使用同样分块平均梯度生成谱初始化。
每项运行恢复其配对 F 实际保存的训练初始 CPU/CUDA RNG，数据采样仍使用 seed+epoch。
固定末轮取分：CoLA MCC×100，MRPC/RTE Accuracy×100；保存诊断 best，不混用。

沿用用户已确认的不同参数预算。F−D、E−C、B−A、D−C、F−E 可在各自相同预算内比较；
D−B 与 (F−E)−(B−A) 同时跨越预算，不能解读成纯分配收益或纯交互。

用户指定在1/2号卡叠加运行，与现有实验并行，每张卡额外最多一个进程。
新训练固定分配：GPU1 20次，GPU2 19次；不等待旧队列结束。
启动前检查空闲显存（CoLA 6GiB / MRPC 9GiB / RTE 14GiB），启用 expandable_segments 减少缓存碎片。
不改变 batch、精度或其他训练超参；并发耗时不用于公平效率对比。
先在1/2号卡运行六个短程验证，全部通过才启动正式39次训练。
发生失败停止启动新任务，让已经启动的本队列任务完成并保留状态及日志。

- `status.json`：GPU、活动任务、步数、待运行项、短程验证和失败记录。
- `results.csv` / `results.json`：54格实时结果与三种子均值、样本标准差。
- `contrasts.json`：预定配对差值，含样本数与预算可比性说明。
- `manifest.json`：全部命令、配置、复用来源、源码与输入记录哈希。
- `checks.json`：CPU上的入口解析、损失公式、模型预算等检查。
- `source_snapshot/`：每项种子实际运行的冻结代码；`references/`：历史结果与随机状态副本。
- `preflight/`：短程验证，与正式训练目录 `runs/` 分开保存。
''')
    publish(manifest, jobs, [], {}, [], [], 'prepared')
    print(json.dumps(dict(root=str(ROOT), cells=54, reused=15, new_runs=39,
        jobs_per_gpu={gpu: sum(j['gpu'] == gpu for j in jobs) for gpu in GPUS}), ensure_ascii=False))


CPU_CHECK = r'''
import inspect, json, runpy, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
assert not torch.cuda.is_available(), "CPU checks must not touch GPUs"
manifest = json.loads(Path(sys.argv[1]).read_text())
runner = sys.argv[2]
namespace = runpy.run_path(runner, run_name="matrix_cpu_check")
jobs = [j for j in manifest["cells"] if j["command"][2] == runner]
for job in jobs:
    sys.argv = job["command"][2:]
    args = namespace["parse_args"]()
    assert args.ablation_variant == job["variant"] and args.seed == job["seed"]
    assert args.rdrop_alpha == 1 and args.epochs == job["expected_hyperparameters"]["epochs"]
    assert args.subspace_consistency_lambda == (job["expected_hyperparameters"]["lambda"] or 0)
from main_ablation_support import variant_spec
labels = torch.tensor([0, 1])
logits1 = torch.tensor([[0.2, -0.4], [0.7, 0.1]], requires_grad=True)
logits2 = torch.tensor([[-0.3, 0.5], [0.1, 0.9]], requires_grad=True)
first = SimpleNamespace(logits=logits1, loss=F.cross_entropy(logits1, labels))
second = SimpleNamespace(logits=logits2, loss=F.cross_entropy(logits2, labels))
loss_fn = namespace["rdrop_loss"]
kwargs = dict(labels=labels, label_smoothing=0.05) if "label_smoothing" in inspect.signature(loss_fn).parameters else {}
loss, ce, kl = loss_fn(first, second, 1.0, **kwargs)
smooth = kwargs.get("label_smoothing", 0.0)
target_ce = 0.5*(F.cross_entropy(logits1, labels, label_smoothing=smooth)+F.cross_entropy(logits2, labels, label_smoothing=smooth))
p, q = logits1.softmax(-1), logits2.softmax(-1)
target_kl = 0.5*((p*(p.log()-q.log())).sum(-1)+(q*(q.log()-p.log())).sum(-1)).mean()
assert torch.allclose(ce, target_ce) and torch.allclose(kl, target_kl, atol=1e-7)
assert torch.allclose(loss, target_ce+target_kl)
loss.backward()
assert torch.isfinite(logits1.grad).all() and torch.isfinite(logits2.grad).all()
for job in jobs:
    lam = 0.5 if job["task"] == "rte" else 1.5
    spec = variant_spec(job["variant"], lam)
    for key in ["init_method", "rank_range", "adapter_parameter_budget", "calibration_steps", "calibration_blocks", "gamma", "lambda"]:
        assert spec[key] == job["expected_hyperparameters"][key], (key, spec, job)
config = json.loads(Path("/workspace/Models/RoBERTa-base/config.json").read_text())
h, f, n = config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]
assert n*(4*2*h+2*(h+f))*8 == 1327104
print(json.dumps({"runner":runner,"parsed_cells":len(jobs),"loss_check":"passed","budget_check":"passed","cuda_visible":False}))
'''


def check():
    manifest = read(ROOT / 'manifest.json')
    rows = []
    for relative, checksum in manifest['source_sha256'].items():
        assert digest(ROOT / relative) == checksum, relative
    for runner in sorted({job['command'][2] for job in manifest['cells']}):
        result = subprocess.run([sys.executable, '-c', CPU_CHECK, str(ROOT / 'manifest.json'), runner],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
            cwd=Path(runner).parents[1], capture_output=True, text=True, timeout=120)
        if result.returncode:
            write(ROOT / 'checks.json', dict(state='failed', runner=runner, stdout=result.stdout, stderr=result.stderr))
            raise RuntimeError(result.stderr or result.stdout)
        rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
    write(ROOT / 'checks.json', dict(state='passed', updated=now(), cpu_checks=rows,
        gpu_smoke_tests='pending; scheduler will execute all six before full training'))
    print(json.dumps(dict(state='passed', runners=len(rows), parsed_cells=sum(row['parsed_cells'] for row in rows))))


def validate(job):
    output = Path(job['output'])
    metric = read(output / 'metrics.json')
    expected = copy.deepcopy(job['expected_hyperparameters'])
    epochs = 1 if job.get('smoke') else expected['epochs']
    if job.get('smoke'):
        expected['scheduler_epochs'] = 1
    assert metric['state'] == 'complete' and metric['seed'] == job['seed'] and metric['task'] == job['task']
    assert metric['visible_gpus'] == str(job['gpu']) and job['gpu'] in GPUS
    assert metric['hyperparameters'] == expected, (metric['hyperparameters'], expected)
    assert metric['epochs'] == metric['final']['epoch'] == epochs
    assert metric['selected'] == metric['final'] and metric['adapter_parameters'] == job['adapter_parameters']
    assert metric['precision'] == 'FP32' and metric['global_batch_size'] == 32
    history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
    assert len(history) == epochs and metric['optimizer_steps'] == history[-1]['step']
    config = read(output / 'run_config.json')
    assert config['source_sha256'] == job['runner_sha256'] and config['smoke'] == bool(job.get('smoke'))
    reference = Path(job['reference'])
    actual_rng, wanted_rng = read(output / 'training_rng.json'), read(reference / 'training_rng.json')
    assert all(actual_rng[key] == wanted_rng[key] for key in ['cpu_sha256', 'cuda_sha256'])
    allocation = read(output / 'allocation.json')
    assert len(allocation['rank_pattern']) == 72
    assert allocation['rank_report']['budget_used'] == job['adapter_parameters']
    initial = read(output / 'initial_state.json')
    assert initial['all_B_zero'] and initial['svd_initialized_modules'] == (72 if job['variant'] in 'BDF' else 0)
    assert initial['adapter_parameter_budget'] == job['adapter_parameters']
    if job['variant'] in 'AB':
        assert set(allocation['rank_pattern'].values()) == {8}
    if job.get('smoke') and job['variant'] == 'A':
        paired_initial = reference / 'reused_A/initial_state.json'
        if paired_initial.exists():
            original_initial = read(paired_initial)
            assert initial['initial_adapter_sha256'] == original_initial['initial_adapter_sha256']
            assert initial['base_classifier_sha256'] == original_initial['base_classifier_sha256']
    assert (output / 'final_adapter/adapter_model.safetensors').is_file()
    return dict(id=job['id'], task=job['task'], seed=job['seed'], variant=job['variant'],
        gpu=job['gpu'], adapter_parameters=job['adapter_parameters'], reused=False,
        output=str(output), metrics=str(output / 'metrics.json'), allocation=str(output / 'allocation.json'),
        score=primary_score(metric), finished=now(), smoke=bool(job.get('smoke')))


def check_pairs(rows):
    for task in TASKS:
        for seed in SEEDS:
            group = {row['variant']: row for row in rows if row['task'] == task and row['seed'] == seed}
            for a, b in [('A', 'B'), ('C', 'D'), ('E', 'F')]:
                if a in group and b in group:
                    assert read(group[a]['allocation'])['rank_pattern'] == read(group[b]['allocation'])['rank_pattern'], (task, seed, a, b)


def publish(manifest, pending, smoke_pending, active, completed, failed, state, waiting=None):
    active_rows = []
    for gpu, (process, job) in active.items():
        status_path = Path(job['output']) / 'status.json'
        active_rows.append(dict(id=job['id'], task=job['task'], seed=job['seed'], variant=job['variant'],
            gpu=gpu, pid=process.pid, smoke=bool(job.get('smoke')), output=job['output'], log=job['log'],
            training_status=read(status_path) if status_path.exists() else {}))
    write(ROOT / 'status.json', dict(state=state, updated=now(), controller_pid=os.getpid(),
        gpus=GPUS, total_cells=54, reused_cells=15, new_runs=39,
        completed_new_runs=sum(not row['smoke'] for row in completed),
        completed_smoke_tests=sum(row['smoke'] for row in completed),
        active=active_rows, pending=[dict(id=j['id'],gpu=j['gpu']) for j in pending],
        smoke_pending=[j['id'] for j in smoke_pending], completed=completed, failed=failed, waiting=waiting or {}))
    available = {row['id']: row for row in manifest['references']}
    available.update({row['id']: row for row in completed if not row['smoke']})
    cells = []
    for cell in manifest['cells']:
        record = available.get(cell['id'])
        cells.append(dict(task=cell['task'], seed=cell['seed'], variant=cell['variant'],
            adapter_parameters=cell['adapter_parameters'], score=record['score'] if record else None,
            reused=bool(record and record['reused']), metrics=record['metrics'] if record else None))
    groups = []
    for task in TASKS:
        for variant in 'ABCDEF':
            values = [row['score'] for row in cells if row['task'] == task and row['variant'] == variant and row['score'] is not None]
            groups.append(dict(task=task, variant=variant, n=len(values), complete=len(values)==3,
                mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None))
    write(ROOT / 'results.json', dict(updated=now(), budget_note=manifest['budget_note'],
        metrics={'cola':'MCC x100','mrpc':'Accuracy x100','rte':'Accuracy x100'}, cells=cells, groups=groups))
    temp = ROOT / 'results.csv.tmp'
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    temp.replace(ROOT / 'results.csv')
    differences = []
    definitions = {'D-B': {'D':1,'B':-1}, 'F-D':{'F':1,'D':-1}, 'E-C':{'E':1,'C':-1},
        'B-A':{'B':1,'A':-1}, 'D-C':{'D':1,'C':-1}, 'F-E':{'F':1,'E':-1},
        '(F-E)-(B-A)':{'F':1,'E':-1,'B':-1,'A':1}}
    for task in TASKS:
        for label, formula in definitions.items():
            paired = []
            for seed in SEEDS:
                group = {row['variant']:row['score'] for row in cells if row['task']==task and row['seed']==seed}
                if all(group[v] is not None for v in formula):
                    paired.append(dict(seed=seed, delta=sum(coef*group[v] for v,coef in formula.items())))
            values = [row['delta'] for row in paired]
            differences.append(dict(task=task, contrast=label, n=len(values), paired=paired,
                budget_controlled=label not in {'D-B','(F-E)-(B-A)'},
                mean_delta=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None))
    write(ROOT / 'contrasts.json', dict(updated=now(), budget_note=manifest['budget_note'], contrasts=differences))


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
    return {int(line.split(',')[0]):int(line.split(',')[1]) for line in output.strip().splitlines()}


def controller():
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(ROOT / 'manifest.json')
        assert read(ROOT / 'checks.json')['state'] == 'passed'
        for relative, checksum in manifest['source_sha256'].items():
            assert digest(ROOT / relative) == checksum, relative
        pending, smoke_pending = list(manifest['jobs']), list(manifest['preflight'])
        active, completed, failed = {}, [], []
        try:
            while pending or smoke_pending or active:
                for gpu, (process, job) in list(active.items()):
                    if process.poll() is None:
                        continue
                    try:
                        if process.returncode:
                            raise RuntimeError(f'exit {process.returncode}; see {job["log"]}')
                        row = validate(job)
                        if row['smoke']:
                            check_pairs([r for r in completed if r['smoke']] + [row])
                        else:
                            check_pairs(manifest['references'] + [r for r in completed if not r['smoke']] + [row])
                        completed.append(row)
                        print(json.dumps({'finished':job['id'],'gpu':gpu,'validated':True}), flush=True)
                    except Exception as error:
                        failed.append(dict(id=job['id'],gpu=gpu,error=repr(error),log=job['log']))
                    del active[gpu]
                if failed and not active:
                    publish(manifest,pending,smoke_pending,active,completed,failed,'failed')
                    return 1
                waiting = {}
                if not failed:
                    free = gpu_memory()
                    # All smoke tests must finish before either GPU starts full training.
                    smoke_stage = bool(smoke_pending) or any(j.get('smoke') for _,j in active.values())
                    for gpu in GPUS:
                        if gpu in active:
                            continue
                        candidates = smoke_pending if smoke_stage else [j for j in pending if j['gpu']==gpu]
                        if not candidates:
                            continue
                        job = next((j for j in candidates if free.get(gpu,0)>=j['min_free_mib']),None)
                        if job is None:
                            waiting[str(gpu)] = dict(reason='insufficient_free_memory',free_mib=free.get(gpu),
                                needed_mib=min(j['min_free_mib'] for j in candidates))
                            continue
                        job = copy.deepcopy(job)
                        job['gpu'] = gpu
                        assert not Path(job['output']).exists()
                        assert digest(job['command'][2]) == job['runner_sha256']
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED=str(job['seed']),
                            CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',
                            HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                        with Path(job['log']).open('x') as stream:
                            process = subprocess.Popen(job['command'],cwd=job['cwd'],env=env,
                                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                        active[gpu] = process, job
                        queue = smoke_pending if job.get('smoke') else pending
                        queue[:] = [item for item in queue if item['id']!=job['id']]
                        print(json.dumps({'started':job['id'],'gpu':gpu,'pid':process.pid}),flush=True)
                state = ('draining_after_failure' if failed else 'preflight' if smoke_pending or any(j.get('smoke') for _,j in active.values())
                         else 'running' if active else 'waiting_for_gpu' if pending else 'complete')
                publish(manifest,pending,smoke_pending,active,completed,failed,state,waiting)
                if pending or smoke_pending or active:
                    time.sleep(10)
        except Exception as error:
            failed.append(dict(error=repr(error)))
            publish(manifest,pending,smoke_pending,active,completed,failed,'controller_error')
            raise
    return 0


def launch():
    assert read(ROOT / 'status.json')['state'] == 'prepared'
    assert read(ROOT / 'checks.json')['state'] == 'passed'
    with (ROOT / 'controller.log').open('x') as stream:
        process = subprocess.Popen([sys.executable,'-u',str(ROOT / 'controller.py'),'--controller'],
            cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT / 'controller_pid.json',dict(pid=process.pid,started=now(),gpus=GPUS))
    print(json.dumps(dict(pid=process.pid,root=str(ROOT),gpus=GPUS,new_runs=39)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--check',action='store_true')
    group.add_argument('--launch',action='store_true')
    group.add_argument('--controller',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.check:
        check()
    elif args.launch:
        launch()
    else:
        sys.exit(controller())
