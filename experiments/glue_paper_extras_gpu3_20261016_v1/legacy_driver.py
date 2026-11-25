#!/usr/bin/env python3
"""Frozen, paired uniform-LoRA/SpecCon x R-Drop experiment on GPU 3."""
import argparse
import ast
import copy
import csv
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

PROJECT = Path('/workspace/SpecCon-LoRA')
ROOT = PROJECT / 'experiments/glue_rdrop_ablation_seed42_gpu3_20261016_v2'
MANIFEST = PROJECT / 'reproduction/current_glue_manifest.json'
TASKS = ['cola', 'mrpc', 'rte']
GPU = 3
SEED = 42
BUDGET = 755712

SUPPORT = '''"""Preparation-only changes; the frozen training loops remain unchanged."""
import hashlib
import json
import time
from pathlib import Path
import torch
from speccon_lora import prepare_specconlora_model, summarize_ranks


def tensor_hash(named):
    result = hashlib.sha256()
    for name, parameter in sorted(named):
        result.update(name.encode())
        result.update(str(tuple(parameter.shape)).encode())
        result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def prepare_ablation_model(model, loader, loss_fn, config, device, method, reference, output):
    start = time.monotonic()
    original = json.loads(Path(reference).read_text())
    classifier_before = tensor_hash([(n, p) for n, p in model.named_parameters() if "classifier" in n])
    if method == "uniform":
        config.adaptive_rank = False
        config.init_method = "none"  # PEFT's Kaiming A and zero B.
        config.r_min = config.r_max = config.base_rank
        config.use_subspace_consistency = False
    model, report = prepare_specconlora_model(model, loader, loss_fn, config, device=device)
    if set(report["target_names"]) != set(original["target_names"]):
        raise RuntimeError("Target modules differ from paired reference")
    if method == "uniform":
        ranks = {name: config.base_rank for name in report["target_names"]}
        report["rank_pattern"] = ranks
        report["rank_summary"] = summarize_ranks(ranks)
        used = report["params"]["trainable_params"]
        report["rank_report"] = {"budget_target": config.param_budget, "budget_used": used,
            "total_adapter_params": used, "budget_unused": config.param_budget - used,
            "rank_min": 8, "rank_max": 8, "rank_mean": 8.0}
        if report["applied_init"]:
            raise RuntimeError("Uniform baseline unexpectedly received spectral initialization")
        if any(value["rank"] != 8 for value in report["applied_scaling"].values()):
            raise RuntimeError("Uniform baseline contains a nonuniform module")
    elif report["rank_pattern"] != original["rank_pattern"]:
        raise RuntimeError("SpecCon rank pattern changed from paired R-Drop=1 reference")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    b_tensors = [(n, p) for n, p in trainable if "lora_B" in n]
    if len(b_tensors) != 72 or any(torch.count_nonzero(p).item() for _, p in b_tensors):
        raise RuntimeError("Expected exactly 72 zero-initialized B matrices")
    if len(trainable) != 144 or sum(p.numel() for _, p in trainable) != config.param_budget:
        raise RuntimeError("Adapter parameter count mismatch")
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError("Non-finite initial parameter")
    payload = {"adapter_method": method, "initial_adapter_sha256": tensor_hash(trainable),
        "base_classifier_sha256": classifier_before, "all_B_zero": True,
        "calibration_performed": method == "speccon", "target_modules": 72,
        "adapter_parameters": config.param_budget, "preparation_seconds": time.monotonic() - start,
        "gs_rank_matches_reference": method == "speccon"}
    Path(output, "initial_state.json").write_text(json.dumps(payload, indent=2) + "\\n")
    print("[AblationInit]", json.dumps(payload), flush=True)
    return model, report
'''


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one occurrence of {old!r}, got {source.count(old)}')
    return source.replace(old, new)


def patch_runner(source):
    result = replace_once(source,
        'from speccon_lora import SpecConLoraConfig, prepare_specconlora_model',
        'from speccon_lora import SpecConLoraConfig, prepare_specconlora_model\n'
        'from ablation_support import prepare_ablation_model')
    result = replace_once(result, '    args = parser.parse_args()\n',
        '    parser.add_argument("--adapter-method", choices=["speccon", "uniform"], required=True)\n'
        '    parser.add_argument("--allocation-reference", type=Path, required=True)\n'
        '    args = parser.parse_args()\n')
    result = replace_once(result, 'def main():\n', 'def main():\n    global PARAM_BUDGET\n')
    main_start = '    args = parse_args(); cfg = dict(TASK_CONFIG[args.task]); output = Path(args.output_dir)\n'
    result = replace_once(result, main_start, main_start +
        '    PARAM_BUDGET = 1_327_104 if args.adapter_method == "uniform" else 755_712\n')
    result = replace_once(result, '    cfg["calibration_steps"] = calibration_steps\n',
        '    cfg["adapter_method"] = args.adapter_method\n'
        '    cfg["adapter_parameter_budget"] = PARAM_BUDGET\n'
        '    cfg["calibration_steps"] = calibration_steps if args.adapter_method == "speccon" else 0\n')
    result = replace_once(result, '    cfg["calibration_blocks"] = 4\n',
        '    cfg["calibration_blocks"] = 4 if args.adapter_method == "speccon" else 0\n')
    result = replace_once(result,
        '    model, report = prepare_specconlora_model(model, calibration_loader, loss_fn, speccon, device=device)',
        '    model, report = prepare_ablation_model(model, calibration_loader, loss_fn, speccon,\n'
        '        device, args.adapter_method, args.allocation_reference, output)')
    result = replace_once(result, '"model": "RoBERTa-base", "method": "SpecCon-LoRA",',
        '"model": "RoBERTa-base", "method": "SpecCon-LoRA" if args.adapter_method == "speccon" else "Uniform-LoRA",')
    result = replace_once(result, '"rank_range": [2, 16], "init_method": "svd_a_zero_b",',
        '"rank_range": [2, 16] if args.adapter_method == "speccon" else [8, 8], "init_method": speccon.init_method,')
    if '"gamma": 1.2, "lambda": 1.5,' in result:
        result = replace_once(result, '"gamma": 1.2, "lambda": 1.5,',
            '"gamma": 1.2 if args.adapter_method == "speccon" else None,\n'
            '                            "lambda": 1.5 if args.adapter_method == "speccon" else None,')
    else:
        result = replace_once(result, '"gamma": args.rank_score_gamma, "lambda": args.subspace_consistency_lambda,',
            '"gamma": args.rank_score_gamma if args.adapter_method == "speccon" else None,\n'
            '                            "lambda": args.subspace_consistency_lambda if args.adapter_method == "speccon" else None,')
    ast.parse(result)
    # A causal experiment must retain the complete optimizer/training/evaluation loop.
    marker = '    epochs = args.smoke_epochs if args.smoke else cfg["epochs"]'
    end = '    result = {\n'
    assert source.split(marker)[1].split(end)[0] == result.split(marker)[1].split(end)[0]
    return result


def set_option(command, option, value):
    if option in command:
        command[command.index(option) + 1] = str(value)
    else:
        command.extend([option, str(value)])


def score(metric):
    key = 'matthews_correlation' if metric['task'] == 'cola' else 'accuracy'
    return 100.0 * metric['final']['splits']['validation'][key]


def prepare():
    records = {row['task']: row for row in read(MANIFEST)['runs']
               if row['seed'] == SEED and row['task'] in TASKS}
    assert set(records) == set(TASKS)
    for task, row in records.items():
        assert digest(row['runner']) == row['runner_sha256']
        source = Path(row['source'])
        metric = read(source / 'metrics.json')
        assert metric['state'] == 'complete' and metric['hyperparameters'] == row['hyperparameters']
        assert metric['adapter_parameters'] == BUDGET and metric['hyperparameters']['rdrop_alpha'] == 1.0
        assert metric['selected'] == metric['final'] and metric['seed'] == SEED
        assert (source / 'final_adapter/adapter_model.safetensors').is_file()
        assert (source / 'training_rng.pt').is_file()
        patch_runner(Path(row['runner']).read_text())
    ast.parse(SUPPORT)
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'logs').mkdir()
    (ROOT / 'preflight').mkdir()
    shutil.copy2(MANIFEST, ROOT / 'original_manifest.json')
    jobs, references = [], []
    for task in TASKS:
        row = records[task]
        source, old_runner = Path(row['source']), Path(row['runner'])
        snapshot = ROOT / 'source_snapshot' / task
        shutil.copytree(Path(row['cwd']) / 'speccon_lora', snapshot / 'speccon_lora',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (snapshot / 'reproduction').mkdir()
        runner = snapshot / 'reproduction' / old_runner.name
        original = old_runner.read_text()
        modified = patch_runner(original)
        runner.write_text(modified)
        (snapshot / 'ablation_support.py').write_text(SUPPORT)
        reference = ROOT / 'references' / task
        reference.mkdir(parents=True)
        for filename in ['metrics.json', 'run_config.json', 'allocation.json', 'training_rng.pt', 'training_rng.json']:
            shutil.copy2(source / filename, reference / filename)
        shutil.copy2(old_runner, reference / 'original_runner.py')
        (reference / 'runner.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), modified.splitlines(True), fromfile=str(old_runner), tofile=str(runner))))
        references.append(dict(task=task, adapter_method='speccon', rdrop_alpha=1.0,
            seed=SEED, reused=True, adapter_parameters=BUDGET, source=str(source), metrics=str(reference / 'metrics.json'),
            score=score(read(reference / 'metrics.json'))))
        # Finish each task's missing three cells before moving to the next task.
        for method, alpha in [('speccon', 0.0), ('uniform', 0.0), ('uniform', 1.0)]:
            job_id = f'{task}_{method}_rd{int(alpha)}_s{SEED}'
            output = ROOT / 'runs' / task / f'{method}_rd{int(alpha)}' / f'seed{SEED}'
            command = list(row['command'])
            command[0] = sys.executable
            command[2] = str(runner)
            for option, value in [('--output-dir', output), ('--adapter-method', method),
                    ('--rdrop-alpha', alpha), ('--training-rng-reference', reference / 'training_rng.pt'),
                    ('--allocation-reference', reference / 'allocation.json')]:
                set_option(command, option, value)
            expected = copy.deepcopy(row['hyperparameters'])
            method_budget = 1327104 if method == 'uniform' else BUDGET
            expected.update(adapter_method=method, rdrop_alpha=alpha,
                adapter_parameter_budget=method_budget,
                training_forwards_per_step=2 if alpha else 1,
                training_rng_reference=str(reference / 'training_rng.pt'))
            if method == 'uniform':
                expected.update(rank_range=[8, 8], init_method='none', gamma=None,
                    **{'lambda': None}, calibration_steps=0, calibration_blocks=0)
            jobs.append(dict(id=job_id, task=task, adapter_method=method, rdrop_alpha=alpha,
                adapter_parameters=method_budget,
                seed=SEED, gpu=GPU, command=command, cwd=str(snapshot), output=str(output),
                log=str(ROOT / 'logs' / (job_id + '.log')), expected_hyperparameters=expected,
                runner_sha256=digest(runner), reference=str(reference)))
    # Exercise all three runners, both preparation branches and both forward-count branches.
    preflight = []
    for task, method, alpha in [('cola', 'uniform', 0.0), ('mrpc', 'uniform', 1.0), ('rte', 'speccon', 0.0)]:
        job = copy.deepcopy(next(j for j in jobs if (j['task'], j['adapter_method'], j['rdrop_alpha']) == (task, method, alpha)))
        job['id'] = 'smoke_' + job['id']
        job['smoke'] = True
        job['output'] = str(ROOT / 'preflight' / job['id'])
        job['log'] = str(ROOT / 'logs' / (job['id'] + '.log'))
        set_option(job['command'], '--output-dir', job['output'])
        job['command'].append('--smoke')
        preflight.append(job)
    shutil.copy2(__file__, ROOT / 'controller.py')
    paths = list((ROOT / 'source_snapshot').rglob('*.py')) + [ROOT / 'controller.py']
    paths += list((ROOT / 'references').rglob('*'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}
    manifest = dict(created=now(), gpu=GPU, seed=SEED, tasks=TASKS, jobs=jobs,
        preflight=preflight, references=references, source_sha256=hashes,
        protocol='User explicitly requested standard uniform rank 8 (1,327,104 parameters) and unchanged SpecCon budget (755,712). Budgets differ across methods but stay fixed within each method. Shared scaling, task recipe and paired training RNG. R-Drop 0 uses one forward; 1 uses two. Fixed final epoch; MRPC accuracy is primary.',
        scheduling='Sequential on GPU 3; require 18 GiB free; do not stop other jobs.')
    write(ROOT / 'manifest.json', manifest)
    write(ROOT / 'status.json', dict(state='prepared', updated=now(), total_new_runs=9, reused_runs=3))
    (ROOT / 'README.md').write_text('''# R-Drop 消融：seed 42，GPU 3

CoLA / MRPC / RTE，uniform LoRA 与完整 SpecCon-LoRA 各比较 R-Drop α=0 / 1。
复用各任务当前保留的 SpecCon+R-Drop α=1、seed42 固定末轮结果；新增 9 次训练。

|任务|训练与调度轮数|学习率|SpecCon λ|label smoothing|LoRA dropout|
|---|---:|---:|---:|---:|---:|
|CoLA|42|1.9e-4|1.5|0|0|
|MRPC|57|3.5e-4|1.5|0.05|0.05|
|RTE|63|2.1e-4|0.5|0|0|

全部 RoBERTa-base、FP32、batch32、72 个相同目标矩阵、缩放 2、A/B 相同学习率。
用户明确指定：LoRA 保持统一 rank=8，共 1,327,104 参数；SpecCon 保持历史预算 755,712 参数。
两种方法之间不是等参数量比较；各自内部 R-Drop 开/关的预算保持一致。
uniform 使用 r=8、PEFT 标准 Kaiming A / 零 B，不做梯度校准；SpecCon 保持原谱初始化、分配和校准数据。
所有新运行恢复各自历史对照实际保存的训练开始 RNG；数据顺序仍由 seed+epoch 的独立 sampler 确定。
α=0 使用一次前向；α=1 使用两次前向及对称 KL。MRPC 四格均保留 label smoothing=0.05。
这是整个 R-Drop 配方的消融，并非等计算量的纯 KL 项消融。GPU 有其他工作，运行时间不用于正式效率比较。

先运行三个短程验证，覆盖三个 runner 和两种 adapter / R-Drop 分支，通过后顺序执行正式训练。
每次启动前检查显存；发生错误停止队列并保存日志，不覆盖已有实验。
CoLA 汇总 MCC×100；MRPC/RTE 汇总 Accuracy×100；另保留 MRPC F1 和组合分数原始记录。

- `manifest.json`：完整命令、冻结配置、原始来源与文件哈希。
- `status.json`：活动进程、训练步数、等待和失败信息。
- `results.json` / `results.csv`：随完成更新的四格表和方法差值。
- `logs/`：独立训练日志；`runs/`：正式输出；`preflight/`：短程验证。
- `references/`：复用结果与 RNG 副本、原始 runner 和变更 diff。
- `source_snapshot/`：本次实际运行代码；历史源码没有改动。

v1 在正式训练前的短程检查中发现统一 rank8 不等于历史预算，已停止。
v2 根据用户确认采用上述不同预算策略，保留 v1 失败日志作为审计记录。
''')
    publish(manifest, [], None, [], [], 'prepared')
    print(json.dumps(dict(root=str(ROOT), jobs=9, reused=3, smoke=3, gpu=GPU), ensure_ascii=False))


def validate(job):
    output = Path(job['output'])
    metric = read(output / 'metrics.json')
    expected = copy.deepcopy(job['expected_hyperparameters'])
    epochs = 1 if job.get('smoke') else expected['epochs']
    if job.get('smoke'):
        expected['scheduler_epochs'] = 1
    assert metric['state'] == 'complete' and metric['seed'] == SEED
    assert metric['task'] == job['task'] and metric['visible_gpus'] == str(GPU)
    assert metric['hyperparameters'] == expected, (metric['hyperparameters'], expected)
    assert metric['epochs'] == epochs and metric['final']['epoch'] == epochs
    assert metric['selected'] == metric['final'] and not metric['stopped_early']
    assert metric['adapter_parameters'] == job['adapter_parameters'] and metric['precision'] == 'FP32'
    assert metric['global_batch_size'] == 32
    history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
    assert len(history) == epochs
    assert metric['optimizer_steps'] == history[-1]['step']
    assert read(output / 'run_config.json')['source_sha256'] == job['runner_sha256']
    assert read(output / 'run_config.json')['smoke'] == bool(job.get('smoke'))
    reference = Path(job['reference'])
    actual_rng, expected_rng = read(output / 'training_rng.json'), read(reference / 'training_rng.json')
    for key in ['cpu_sha256', 'cuda_sha256']:
        assert actual_rng[key] == expected_rng[key]
    allocation = read(output / 'allocation.json')
    assert allocation['rank_report']['budget_used'] == job['adapter_parameters']
    assert len(allocation['rank_pattern']) == 72
    if job['adapter_method'] == 'speccon':
        assert allocation['rank_pattern'] == read(reference / 'allocation.json')['rank_pattern']
    else:
        assert set(allocation['rank_pattern'].values()) == {8}
    initial = read(output / 'initial_state.json')
    assert initial['all_B_zero'] and initial['adapter_parameters'] == job['adapter_parameters']
    if job['adapter_method'] == 'uniform' and job['rdrop_alpha'] == 1 and not job.get('smoke'):
        paired = ROOT / 'runs' / job['task'] / 'uniform_rd0' / f'seed{SEED}' / 'initial_state.json'
        other = read(paired)
        assert initial['initial_adapter_sha256'] == other['initial_adapter_sha256']
        assert initial['base_classifier_sha256'] == other['base_classifier_sha256']
    if job['rdrop_alpha'] == 0:
        assert all(row['train_kl_loss'] == 0 for row in history)
    assert (output / 'final_adapter/adapter_model.safetensors').is_file()
    return dict(id=job['id'], task=job['task'], adapter_method=job['adapter_method'],
        adapter_parameters=job['adapter_parameters'],
        rdrop_alpha=job['rdrop_alpha'], seed=SEED, reused=False, score=score(metric),
        output=str(output), metrics=str(output / 'metrics.json'), finished=now(), smoke=bool(job.get('smoke')))


def publish(manifest, pending, active, completed, failed, state):
    active_row = None
    if active:
        process, job = active
        status = Path(job['output']) / 'status.json'
        active_row = dict(id=job['id'], task=job['task'], adapter_method=job['adapter_method'],
            rdrop_alpha=job['rdrop_alpha'], pid=process.pid, smoke=bool(job.get('smoke')),
            output=job['output'], log=job['log'], training_status=read(status) if status.exists() else {})
    write(ROOT / 'status.json', dict(state=state, updated=now(), gpu=GPU, seed=SEED,
        controller_pid=os.getpid(), total_new_runs=9, reused_runs=3, active=active_row,
        pending=[job['id'] for job in pending], completed=completed, failed=failed))
    rows = list(manifest['references']) + [row for row in completed if not row['smoke']]
    cells = []
    for task in TASKS:
        for method in ['uniform', 'speccon']:
            for alpha in [0.0, 1.0]:
                row = next((r for r in rows if (r['task'], r['adapter_method'], r['rdrop_alpha']) == (task, method, alpha)), None)
                cells.append(dict(task=task, adapter_method=method, rdrop_alpha=alpha, seed=SEED,
                    adapter_parameters=1327104 if method == 'uniform' else BUDGET,
                    score=row['score'] if row else None, reused=row['reused'] if row else False,
                    metrics=row['metrics'] if row else None))
    effects = []
    for task in TASKS:
        def get(method, alpha):
            return next(r['score'] for r in cells if (r['task'], r['adapter_method'], r['rdrop_alpha']) == (task, method, alpha))
        u0, u1, g0, g1 = get('uniform', 0), get('uniform', 1), get('speccon', 0), get('speccon', 1)
        effects.append(dict(task=task,
            speccon_minus_uniform_without_rdrop=g0-u0 if g0 is not None and u0 is not None else None,
            speccon_minus_uniform_with_rdrop=g1-u1 if g1 is not None and u1 is not None else None,
            rdrop_gain_uniform=u1-u0 if u1 is not None and u0 is not None else None,
            rdrop_gain_speccon=g1-g0 if g1 is not None and g0 is not None else None,
            interaction=(g1-u1)-(g0-u0) if all(v is not None for v in [u0,u1,g0,g1]) else None))
    write(ROOT / 'results.json', dict(updated=now(), seed=SEED,
        budget_note='User-selected unequal budgets: uniform LoRA r8=1327104; SpecCon=755712. Within-method R-Drop comparisons keep budget fixed.',
        metrics={'cola': 'MCC x100', 'mrpc': 'Accuracy x100', 'rte': 'Accuracy x100'}, cells=cells, effects=effects))
    temporary = ROOT / 'results.csv.tmp'
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    temporary.replace(ROOT / 'results.csv')


def controller():
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(ROOT / 'manifest.json')
        for relative, checksum in manifest['source_sha256'].items():
            assert digest(ROOT / relative) == checksum, relative
        pending = list(manifest['preflight']) + list(manifest['jobs'])
        completed, failed, active = [], [], None
        try:
            while pending or active:
                if active and active[0].poll() is not None:
                    process, job = active
                    try:
                        if process.returncode:
                            raise RuntimeError(f'exit {process.returncode}; see {job["log"]}')
                        completed.append(validate(job))
                        print(json.dumps({'finished': job['id'], 'validated': True}), flush=True)
                    except Exception as error:
                        failed.append(dict(id=job['id'], error=repr(error), log=job['log']))
                    active = None
                if failed:
                    publish(manifest, pending, active, completed, failed, 'failed')
                    return 1
                if not active and pending:
                    free = int(subprocess.check_output(['nvidia-smi', '-i', str(GPU),
                        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                    if free < 18432:
                        publish(manifest, pending, active, completed, failed, 'waiting_for_gpu')
                        time.sleep(10)
                        continue
                    job = pending[0]
                    assert not Path(job['output']).exists()
                    assert digest(job['command'][2]) == job['runner_sha256']
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), PYTHONHASHSEED=str(SEED),
                        CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
                        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                    with Path(job['log']).open('x') as stream:
                        process = subprocess.Popen(job['command'], cwd=job['cwd'], env=env,
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    active = process, job
                    pending.pop(0)
                    print(json.dumps({'started': job['id'], 'gpu': GPU, 'pid': process.pid}), flush=True)
                state = 'preflight' if active and active[1].get('smoke') else ('running' if active or pending else 'complete')
                publish(manifest, pending, active, completed, failed, state)
                if active or pending:
                    time.sleep(10)
        except Exception as error:
            failed.append(dict(error=repr(error)))
            publish(manifest, pending, active, completed, failed, 'controller_error')
            raise
    return 0


def launch():
    assert read(ROOT / 'status.json')['state'] == 'prepared'
    with (ROOT / 'controller.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'controller.py'), '--controller'],
            cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(ROOT / 'controller_pid.json', dict(pid=process.pid, started=now()))
    print(json.dumps(dict(pid=process.pid, root=str(ROOT), gpu=GPU)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--launch', action='store_true')
    group.add_argument('--controller', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        launch()
    else:
        sys.exit(controller())
