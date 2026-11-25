#!/usr/bin/env python3
"""Sequential GLUE lambda=0 ablation, seed 42, on GPU 2."""
import argparse
import ast
import copy
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
ROOT = PROJECT / 'experiments/glue_lambda0_seed42_gpu2_20261015_v1'
BASE_MANIFEST = PROJECT / 'reproduction/current_glue_manifest.json'
NEW_BASELINE = PROJECT / 'experiments/rte_qnli_lambda15_3seeds_gpu012_20261015_v1'
TASKS = ['rte', 'mrpc', 'stsb', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']
GPU = 2


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_option(command, option, value):
    if option in command:
        command[command.index(option) + 1] = str(value)
    else:
        command.extend([option, str(value)])


def expose_lambda(original):
    """Add the existing lambda interface to older frozen runners, without changing defaults."""
    if 'parser.add_argument("--subspace-consistency-lambda"' in original:
        return original
    changes = [
        ('    args = parser.parse_args()\n',
         '    parser.add_argument("--subspace-consistency-lambda", type=float, default=1.5)\n'
         '    args = parser.parse_args()\n'
         '    if not math.isfinite(args.subspace_consistency_lambda) or args.subspace_consistency_lambda < 0:\n'
         '        parser.error("--subspace-consistency-lambda must be finite and nonnegative")\n'),
        ('subspace_consistency_lambda=1.5,',
         'subspace_consistency_lambda=args.subspace_consistency_lambda,'),
        ('"gamma": 1.2, "lambda": 1.5,',
         '"gamma": 1.2, "lambda": args.subspace_consistency_lambda,'),
        ('        "smoke": args.smoke, "scheduler_epochs": cfg["scheduler_epochs"],\n',
         '        "smoke": args.smoke, "scheduler_epochs": cfg["scheduler_epochs"],\n'
         '        "subspace_consistency_lambda": args.subspace_consistency_lambda,\n'),
    ]
    result = original
    for before, after in changes:
        assert result.count(before) == 1, before
        result = result.replace(before, after)
    ast.parse(result)
    return result


def prepare():
    runs = {r['task']: r for r in read(BASE_MANIFEST)['runs'] if r['seed'] == 42}
    assert set(runs) == set(TASKS)
    # Validate all inputs before creating the campaign.
    for task in TASKS:
        run = runs[task]
        assert digest(run['runner']) == run['runner_sha256']
        source = Path(run['source'])
        assert (source / 'training_rng.pt').is_file()
        assert read(source / 'metrics.json')['hyperparameters'] == run['hyperparameters']
        expose_lambda(Path(run['runner']).read_text())
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'logs').mkdir()
    shutil.copy2(BASE_MANIFEST, ROOT / 'original_manifest.json')
    jobs = []
    for task in TASKS:
        run = runs[task]
        source, original_runner = Path(run['source']), Path(run['runner'])
        snapshot = ROOT / 'source_snapshot' / task
        shutil.copytree(Path(run['cwd']) / 'speccon_lora', snapshot / 'speccon_lora',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (snapshot / 'reproduction').mkdir()
        runner = snapshot / 'reproduction' / original_runner.name
        original = original_runner.read_text()
        modified = expose_lambda(original)
        runner.write_text(modified)
        reference = ROOT / 'references' / task
        reference.mkdir(parents=True)
        shutil.copy2(original_runner, reference / 'original_runner.py')
        (reference / 'runner.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), modified.splitlines(True),
            fromfile=str(original_runner), tofile=str(runner))))
        for filename in ('run_config.json', 'metrics.json', 'training_rng.pt', 'training_rng.json'):
            shutil.copy2(source / filename, reference / filename)
        rng = reference / 'training_rng.pt'
        output = ROOT / 'runs' / task / 'seed42'
        command = list(run['command'])
        command[2] = str(runner)
        set_option(command, '--output-dir', output)
        set_option(command, '--subspace-consistency-lambda', 0.0)
        # Restore each baseline's actual training RNG after rank-dependent initialization.
        set_option(command, '--training-rng-reference', rng)
        expected = copy.deepcopy(read(source / 'run_config.json'))
        expected['source_sha256'] = digest(runner)
        expected['subspace_consistency_lambda'] = 0.0
        expected['config']['training_rng_reference'] = str(rng)
        expected_hp = dict(run['hyperparameters'], **{'lambda': 0.0, 'training_rng_reference': str(rng)})
        baseline_metrics = (NEW_BASELINE / 'runs' / task / 'seed42' / 'metrics.json'
                            if task in ('rte', 'qnli') else reference / 'metrics.json')
        jobs.append(dict(task=task, seed=42, gpu=GPU, command=command, cwd=str(snapshot),
            output=str(output), log=str(ROOT / 'logs' / f'{task}_seed42.log'),
            expected_run_config=expected, expected_hyperparameters=expected_hp,
            baseline_metrics=str(baseline_metrics),
            baseline_lambda=1.5 if task in ('rte', 'qnli') else run['hyperparameters']['lambda'],
            original_source=str(source), original_runner_sha256=run['runner_sha256'],
            rng_sha256=digest(rng)))
    shutil.copy2(__file__, ROOT / 'controller.py')
    protected = list((ROOT / 'source_snapshot').rglob('*.py')) + [ROOT / 'controller.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    write(ROOT / 'manifest.json', dict(created=now(), gpu=GPU, seeds=[42], tasks=TASKS,
          **{'lambda': 0.0}, selection='fixed final epoch', jobs=jobs, source_sha256=hashes,
          rng_protocol='Restore the exact CPU/CUDA training RNG saved by the retained seed42 run.',
          scheduling='One dataset at a time on GPU 2, short tasks first.'))
    write(ROOT / 'status.json', dict(state='prepared', updated=now(), gpu=GPU, total=8,
          active=None, pending=TASKS, completed=[], failed=[]))
    rows = ['# GLUE λ=0 消融队列', '',
            'GPU 2，每次运行一个数据集，仅 seed 42。全部任务 λ=0，其他训练超参数沿用当前保留配置。', '',
            '|顺序|任务|训练/调度轮数|对照 λ|', '|---:|---|---:|---:|']
    for index, job in enumerate(jobs, 1):
        rows.append(f'|{index}|{job["task"]}|{job["expected_hyperparameters"]["epochs"]}|{job["baseline_lambda"]}|')
    rows += ['', 'RTE、QNLI 与本轮 λ=1.5 的 seed42 对照；其余任务与当前报告的 seed42 对照。'
             'QNLI 原报告已有 λ=0，本队列按请求重新训练，待 λ=1.5 对照完成后计算差值。', '',
             '各任务恢复原训练开始时的 CPU/CUDA 随机状态，以控制 rank 变化引起的初始化随机数消耗差异。'
             '仅复用随机状态，不复用训练权重。旧源码未提供 λ 参数的，仅在本次快照中增加该参数、校验和记录；'
             '默认仍为 1.5，源码差异保存在 references/<task>/runner.patch。', '',
             '`status.json` 每 10 秒记录进度、PID 和等待队列；`logs/` 为逐任务日志；'
             '`runs/` 保存训练结果；`results.json` 记录末轮分数与配对对照。原有 0/1 号卡队列继续运行。', '',
             '后台启动入口：`/root/anaconda3/bin/python reproduction/launch_glue_lambda0_seed42_gpu2_20261015.py --launch`。'
             '同一目录拒绝重复启动；发生失败时保留日志并停止后续任务。']
    (ROOT / 'README.md').write_text('\n'.join(rows) + '\n')
    print(json.dumps({'prepared': str(ROOT), 'jobs': len(jobs), 'gpu': GPU, 'order': TASKS}))


def validate(job):
    output = Path(job['output'])
    assert read(output / 'run_config.json') == job['expected_run_config']
    metric = read(output / 'metrics.json')
    assert metric['state'] == 'complete' and metric['task'] == job['task'] and metric['seed'] == 42
    assert metric['hyperparameters'] == job['expected_hyperparameters']
    assert metric['precision'] == 'FP32' and metric['global_batch_size'] == 32
    assert metric['adapter_parameters'] == 755712 and metric['visible_gpus'] == str(GPU)
    assert metric['epochs'] == job['expected_hyperparameters']['epochs']
    assert metric['selected'] == metric['final'] and metric['final']['epoch'] == metric['epochs']
    assert len((output / 'history.jsonl').read_text().splitlines()) == metric['epochs']
    actual_rng = read(output / 'training_rng.json')
    expected_rng = read(ROOT / 'references' / job['task'] / 'training_rng.json')
    for key in ('cpu_sha256', 'cuda_sha256'):
        assert actual_rng[key] == expected_rng[key]
    return dict(task=job['task'], seed=42, score=metric['final_score_percent'],
                output=str(output), finished=now())


def publish(jobs, pending, active, completed, failed, state=None):
    active_row = None
    if active:
        process, job = active
        status_path = Path(job['output']) / 'status.json'
        active_row = dict(task=job['task'], seed=42, gpu=GPU, pid=process.pid,
                          output=job['output'], log=job['log'],
                          training_status=read(status_path) if status_path.exists() else {})
    write(ROOT / 'status.json', dict(state=state or ('running' if active or pending else 'complete'),
          updated=now(), controller_pid=os.getpid(), gpu=GPU, total=8,
          active=active_row, pending=[j['task'] for j in pending], completed=completed, failed=failed))
    results = []
    for row in completed:
        job = next(j for j in jobs if j['task'] == row['task'])
        baseline = Path(job['baseline_metrics'])
        result = dict(row, baseline_lambda=job['baseline_lambda'], baseline_metrics=str(baseline),
                      baseline_score=None, delta=None)
        if baseline.is_file():
            metric = read(baseline)
            if metric['state'] == 'complete' and metric['hyperparameters']['lambda'] == job['baseline_lambda']:
                result['baseline_score'] = metric['final_score_percent']
                result['delta'] = row['score'] - result['baseline_score']
        results.append(result)
    write(ROOT / 'results.json', dict(updated=now(), seed=42, runs=results))


def controller():
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(ROOT / 'manifest.json')
        for relative, checksum in manifest['source_sha256'].items():
            assert digest(ROOT / relative) == checksum, relative
        jobs = manifest['jobs']
        pending, completed, failed, active = list(jobs), [], [], None
        try:
            while pending or active:
                if active and active[0].poll() is not None:
                    process, job = active
                    try:
                        if process.returncode:
                            raise RuntimeError(f'exit {process.returncode}; see {job["log"]}')
                        completed.append(validate(job))
                    except Exception as error:
                        failed.append(dict(task=job['task'], error=repr(error), log=job['log']))
                    print(json.dumps({'finished': job['task'], 'exit_code': process.returncode}), flush=True)
                    active = None
                if failed:
                    publish(jobs, pending, active, completed, failed, state='failed')
                    return 1
                if not active and pending:
                    free = int(subprocess.check_output(['nvidia-smi', '-i', str(GPU),
                        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                    if free < 20000:
                        publish(jobs, pending, active, completed, failed, state='waiting_for_gpu')
                        time.sleep(10)
                        continue
                    job = pending[0]
                    assert not Path(job['output']).exists()
                    assert digest(ROOT / 'references' / job['task'] / 'training_rng.pt') == job['rng_sha256']
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), PYTHONHASHSEED='42',
                        CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
                        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                    with Path(job['log']).open('x') as stream:
                        process = subprocess.Popen(job['command'], cwd=job['cwd'], env=env,
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    active = (process, job)
                    pending.pop(0)
                    print(json.dumps({'started': job['task'], 'seed': 42, 'gpu': GPU, 'pid': process.pid}), flush=True)
                publish(jobs, pending, active, completed, failed)
                if active or pending:
                    time.sleep(10)
        except Exception as error:
            failed.append(dict(error=repr(error)))
            publish(jobs, pending, active, completed, failed, state='controller_error')
            raise
    return 0


def launch():
    assert read(ROOT / 'status.json')['state'] == 'prepared'
    with (ROOT / 'controller.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'controller.py'), '--controller'],
            cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(ROOT / 'controller_pid.json', dict(pid=process.pid, started=now()))
    print(json.dumps({'pid': process.pid, 'root': str(ROOT), 'gpu': GPU}))


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
