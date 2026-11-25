#!/usr/bin/env python3
"""One-at-a-time GPU 3 queue for the preregistered P1-A extension."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXP = Path('/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1')
MAIN = Path('/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3')
RUNS = EXP / 'runs/p1a_gpu3_extension_20261106'
LOGS = EXP / 'logs'
STATUS = EXP / 'status_p1a_gpu3_extension.json'
CONTROL_LOG = EXP / 'controller_p1a_gpu3_extension.log'
JOBS = [('rte', 20261106, seed) for seed in (42, 777, 2026)] + [
    (task, 20261107, seed) for task in ('mrpc', 'cola') for seed in (42, 777, 2026)]
CONFIG = {
    'rte': ('0.00021', '63', '0.198', '0.0', '0.5'),
    'mrpc': ('0.00035', '57', '0.307', '0.05', '1.5'),
    'cola': ('0.00019', '42', '0.196', '0.0', '1.5'),
}


def note(msg):
    line = time.strftime('%Y-%m-%dT%H:%M:%S%z') + ' ' + msg
    with CONTROL_LOG.open('a') as stream:
        stream.write(line + '\n')
    print(line, flush=True)


def save(entries):
    tmp = STATUS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                               'controller_pid': os.getpid(), 'entries': entries}, indent=2) + '\n')
    tmp.replace(STATUS)


def gpu_admitted():
    samples = []
    for _ in range(3):
        text = subprocess.check_output(['nvidia-smi', '-i', '3',
                                        '--query-gpu=memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
        free, util = (int(v) for v in text.strip().split(','))
        samples.append((free, util))
        time.sleep(2)
    return min(free for free, _ in samples) >= 9 * 1024 and sum(util for _, util in samples) / 3 <= 75


def launch(task, shuffle_seed, seed):
    source = EXP / 'p1a_source' / task
    runner = source / 'reproduction' / ('run_roberta_cola_rdrop.py' if task == 'cola' else 'run_roberta_small_glue_rdrop.py' if task == 'rte' else 'run_roberta_small_glue_regularization.py')
    ref = MAIN / 'references' / task / f'seed{seed}'
    out = RUNS / task / f'shuffle{shuffle_seed}' / f'seed{seed}'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Refusing nonempty output: {out}')
    for path in (runner, ref / 'training_rng.pt', ref / 'reused_F/allocation.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    lr, epochs, wd, dropout, lam = CONFIG[task]
    cmd = [sys.executable, '-u', str(runner), '--task', task, '--seed', str(seed),
           '--output-dir', str(out), '--learning-rate', lr, '--epochs', epochs,
           '--scheduler-epochs', epochs, '--weight-decay', wd,
           '--lora-dropout', dropout, '--lr-scheduler', 'linear',
           '--calibration-steps', '16', '--rdrop-alpha', '1.0',
           '--subspace-consistency-lambda', lam, '--ablation-variant', 'F',
           '--allocation-reference', str(ref / 'reused_F/allocation.json'),
           '--training-rng-reference', str(ref / 'training_rng.pt')]
    if task == 'mrpc':
        cmd += ['--label-smoothing', '0.05']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', PYTHONHASHSEED=str(seed),
               SPECCON_SHUFFLE_SEED=str(shuffle_seed),
               SPECCON_SHUFFLE_MANIFEST=str(out / 'shuffle_manifest.json'))
    log = LOGS / f'p1a_gpu3_{task}_shuffle{shuffle_seed}_seed{seed}.log'
    with log.open('ab') as stream:
        proc = subprocess.Popen(cmd, cwd=source, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    note(f'START task={task} shuffle={shuffle_seed} seed={seed} gpu=3 pid={proc.pid}')
    return proc, out, log


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        entries = json.loads(STATUS.read_text())['entries']
        if [(x['task'], x['shuffle_seed'], x['seed']) for x in entries] != JOBS:
            raise RuntimeError('Queue differs from preregistration')
        for x in entries:
            out = Path(x['output'])
            try:
                metrics = json.loads((out / 'metrics.json').read_text())
            except (OSError, json.JSONDecodeError):
                metrics = {}
            if metrics.get('state') == 'complete' and metrics.get('adapter_parameters') == 755712:
                x['state'] = 'complete'
            elif x['state'] == 'running':
                raise RuntimeError(f'Stale running task lacks valid metrics: {out}')
    else:
        entries = [{'task': task, 'shuffle_seed': shuf, 'seed': seed, 'state': 'queued',
                    'output': str(RUNS / task / f'shuffle{shuf}' / f'seed{seed}')}
                   for task, shuf, seed in JOBS]
    note(f'START controller pid={os.getpid()} jobs={len(JOBS)} gpu=3')
    save(entries)
    active = None
    failed = False
    while True:
        if active is not None:
            entry, proc, out = active
            code = proc.poll()
            if code is not None:
                active = None
                try:
                    metrics = json.loads((out / 'metrics.json').read_text())
                except (OSError, json.JSONDecodeError):
                    metrics = {}
                ok = code == 0 and metrics.get('state') == 'complete' and metrics.get('adapter_parameters') == 755712
                entry.update(state='complete' if ok else 'failed', exit_code=code)
                failed |= not ok
                note(f'EXIT {entry["task"]} shuffle={entry["shuffle_seed"]} seed={entry["seed"]} state={entry["state"]} code={code}')
        if active is None and not failed:
            queued = next((x for x in entries if x['state'] == 'queued'), None)
            if queued and gpu_admitted():
                try:
                    proc, out, log = launch(queued['task'], queued['shuffle_seed'], queued['seed'])
                    queued.update(state='running', gpu=3, pid=proc.pid, log=str(log))
                    active = (queued, proc, out)
                except Exception as exc:
                    queued.update(state='failed', error=str(exc))
                    failed = True
                    note(f'LAUNCH FAILED: {exc}')
        save(entries)
        if active is None and (failed or all(x['state'] == 'complete' for x in entries)):
            note('STOP failed=' + str(failed))
            return 1 if failed else 0
        time.sleep(30)


if __name__ == '__main__':
    sys.exit(main())
