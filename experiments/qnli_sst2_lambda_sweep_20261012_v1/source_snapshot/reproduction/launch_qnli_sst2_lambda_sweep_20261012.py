#!/usr/bin/env python3
"""Full-schedule paired lambda sweep for QNLI and SST-2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime

PROJECT = Path('/workspace/SpecCon-LoRA')
BASE = PROJECT / 'experiments/glue_rdrop_best_seed2026_gpu01_20261008_v1'
ROOT = PROJECT / 'experiments/qnli_sst2_lambda_sweep_20261012_v1'
SEEDS = [42, 777, 2026]
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.5]
TASKS = {
    'qnli': dict(epochs=49, lr=4.8e-4, wd=0.283, dropout=0.05),
    'sst2': dict(epochs=38, lr=2.0e-4, wd=0.482, dropout=0.05),
}


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


def label(value):
    return str(value).replace('.', 'p')


def jobs():
    return [dict(task=task, seed=seed, lam=lam)
            for lam in LAMBDAS for seed in SEEDS for task in TASKS]


def prepare():
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'logs').mkdir()
    snapshot = ROOT / 'source_snapshot'
    shutil.copytree(BASE / 'source_snapshot/speccon_lora', snapshot / 'speccon_lora',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (snapshot / 'reproduction').mkdir()
    for filename in ['run_roberta_other_glue_rdrop.py', Path(__file__).name]:
        shutil.copy2(PROJECT / 'reproduction' / filename, snapshot / 'reproduction' / filename)
    groups = {group['task']: group for group in read(BASE / 'summary.json')['groups']}
    references = {}
    for task, cfg in TASKS.items():
        assert len(groups[task]['per_seed']) == len(SEEDS)
        for item in groups[task]['per_seed']:
            source = Path(item['source'])
            metric = read(source / 'metrics.json')
            hp = metric['hyperparameters']
            assert metric['state'] == 'complete' and metric['task'] == task
            assert metric['seed'] == item['seed'] and metric['epochs'] == cfg['epochs']
            assert hp['lr'] == cfg['lr'] and hp['scheduler_epochs'] == cfg['epochs']
            assert hp['wd'] == cfg['wd'] and hp['dropout'] == cfg['dropout']
            assert hp['gamma'] == 1.2 and hp['lambda'] == 1.5
            assert hp['rdrop_alpha'] == 1.0 and hp['calibration_steps'] == 16
            assert metric['adapter_parameters'] == 755712
            score = 100 * metric['final']['splits']['validation']['accuracy']
            assert abs(score - item['score']) < 1e-8
            references[f'{task}:{item["seed"]}'] = dict(score=score, source=str(source),
                rng=str(source / 'training_rng.pt'))
    assert len(references) == 6
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in snapshot.rglob('*.py')}
    write(ROOT / 'manifest.json', dict(created=now(), task_config=TASKS, seeds=SEEDS,
        baseline_lambda=1.5, candidate_lambdas=LAMBDAS, jobs=jobs(), references=references,
        fixed=dict(model='RoBERTa-base', gamma=1.2, rdrop_alpha=1.0,
                   calibration_steps=16, parameter_budget=755712,
                   scheduler='linear', selection='fixed final epoch validation accuracy'),
        source_sha256=hashes))
    write(ROOT / 'status.json', dict(state='prepared', updated=now(), total=len(jobs()),
                                     completed=0, active=[], pending=jobs(), failed=[]))


def gpu_free_and_util(index):
    output = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return [int(value.strip()) for value in output.strip().split(',')]


def launch(job, gpu, reference):
    task, seed, lam = job['task'], job['seed'], job['lam']
    cfg = TASKS[task]
    output = ROOT / 'runs' / task / f'lambda{label(lam)}' / f'seed{seed}'
    log = ROOT / 'logs' / f'{task}_lambda{label(lam)}_seed{seed}.log'
    assert not output.exists() and not log.exists()
    command = [sys.executable, '-u', str(ROOT / 'source_snapshot/reproduction/run_roberta_other_glue_rdrop.py'),
        '--task', task, '--seed', str(seed), '--output-dir', str(output),
        '--epochs', str(cfg['epochs']), '--scheduler-epochs', str(cfg['epochs']),
        '--learning-rate', str(cfg['lr']), '--weight-decay', str(cfg['wd']),
        '--lora-dropout', str(cfg['dropout']), '--lr-scheduler', 'linear',
        '--calibration-steps', '16', '--rdrop-alpha', '1.0',
        '--subspace-consistency-lambda', str(lam),
        '--training-rng-reference', reference['rng']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(seed),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT / 'source_snapshot', env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    return process, dict(job, gpu=gpu, pid=process.pid, output=str(output),
                         log=str(log), started=now())


def validate(info, reference):
    metric = read(Path(info['output']) / 'metrics.json')
    task, seed, lam = info['task'], info['seed'], info['lam']
    cfg = TASKS[task]
    hp = metric['hyperparameters']
    assert metric['state'] == 'complete' and metric['task'] == task
    assert metric['seed'] == seed and metric['epochs'] == cfg['epochs']
    assert metric['selected'] == metric['final'] and metric['final']['epoch'] == cfg['epochs']
    assert hp['lr'] == cfg['lr'] and hp['scheduler_epochs'] == cfg['epochs']
    assert hp['wd'] == cfg['wd'] and hp['dropout'] == cfg['dropout']
    assert hp['gamma'] == 1.2 and hp['lambda'] == lam
    assert hp['rdrop_alpha'] == 1.0 and metric['adapter_parameters'] == 755712
    assert len((Path(info['output']) / 'history.jsonl').read_text().splitlines()) == cfg['epochs']
    score = 100 * metric['final']['splits']['validation']['accuracy']
    return dict(task=task, seed=seed, lam=lam, score=score,
                baseline=reference['score'], delta=score-reference['score'],
                output=info['output'], gpu=info['gpu'])


def report(done, active, pending, failed):
    write(ROOT / 'results.json', dict(updated=now(), runs=done))
    write(ROOT / 'status.json', dict(updated=now(), total=len(jobs()), completed=len(done),
        state='running' if active else 'failed' if failed else 'waiting_for_gpu' if pending else 'complete',
        active=[dict(info, training_status=read(Path(info['output']) / 'status.json')
                     if (Path(info['output']) / 'status.json').exists() else {})
                for _, info in active], pending=pending, failed=failed))
    lines = ['# QNLI / SST-2 lambda 搜索', '',
        '固定各任务原训练轮数、线性调度和其余参数；主指标为最终轮验证集准确率。',
        '原配置 lambda=1.5；每个候选值按 seed 42、777、2026 与原配置配对。', '',
        '|任务|lambda|完成 seed|新配置均值|配对基线均值|差值|',
        '|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        for lam in LAMBDAS:
            rows = [row for row in done if row['task'] == task and row['lam'] == lam]
            if not rows:
                continue
            count = len(rows)
            score = statistics.mean(row['score'] for row in rows)
            baseline = statistics.mean(row['baseline'] for row in rows)
            lines.append(f'|{task.upper()}|{lam:g}|{count}/3|{score:.2f}|{baseline:.2f}|{score-baseline:+.2f}|')
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def controller():
    manifest = read(ROOT / 'manifest.json')
    for filename, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'source_snapshot' / filename).read_bytes()).hexdigest() == digest
    pending, active, done, failed = list(manifest['jobs']), [], [], []
    while pending or active:
        still = []
        for process, info in active:
            code = process.poll()
            if code is None:
                still.append((process, info))
                continue
            try:
                if code:
                    raise RuntimeError(f'exit {code}; see {info["log"]}')
                done.append(validate(info, manifest['references'][f'{info["task"]}:{info["seed"]}']))
            except Exception as error:
                failed.append(dict(info, error=repr(error)))
        active = still
        if not failed:
            for gpu in [0, 2, 1, 3]:
                if not pending or any(info['gpu'] == gpu for _, info in active):
                    continue
                free, utilization = gpu_free_and_util(gpu)
                if free >= 14000 and utilization <= 70:
                    job = pending.pop(0)
                    active.append(launch(job, gpu, manifest['references'][f'{job["task"]}:{job["seed"]}']))
        report(done, active, pending, failed)
        if failed and not active:
            return 1
        if pending or active:
            time.sleep(30)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.controller:
        sys.exit(controller())
    if args.launch:
        assert read(ROOT / 'status.json')['state'] == 'prepared'
        with (ROOT / 'controller.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(__file__), '--controller'],
                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        write(ROOT / 'controller_pid.json', dict(pid=process.pid, started=now()))
        print(json.dumps(dict(pid=process.pid, root=str(ROOT))))
