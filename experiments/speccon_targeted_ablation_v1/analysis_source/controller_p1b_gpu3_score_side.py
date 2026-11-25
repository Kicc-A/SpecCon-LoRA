#!/usr/bin/env python3
"""Wait for P1-A, then run right and joint consensus controls on GPU 3."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from controller_p1a_gpu3_extension import gpu_admitted

EXP = Path('/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1')
MAIN = Path('/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3')
GATE = EXP / 'status_p1a_gpu3_extension.json'
RUNS = EXP / 'runs/p1b_gpu3_score_side_20261106'
LOGS = EXP / 'logs'
STATUS = EXP / 'status_p1b_gpu3_score_side.json'
CONTROL_LOG = EXP / 'controller_p1b_gpu3_score_side.log'
JOBS = [(task, side, seed) for task in ('mrpc', 'cola')
        for side in ('right', 'joint') for seed in (42, 777, 2026)]


def note(msg):
    line = time.strftime('%Y-%m-%dT%H:%M:%S%z') + ' ' + msg
    with CONTROL_LOG.open('a') as stream:
        stream.write(line + '\n')
    print(line, flush=True)


def save(entries, stage):
    tmp = STATUS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                               'controller_pid': os.getpid(), 'stage': stage,
                               'entries': entries}, indent=2) + '\n')
    tmp.replace(STATUS)


def gate_state():
    if not GATE.exists():
        return 'waiting'
    entries = json.loads(GATE.read_text())['entries']
    if any(x['state'] == 'failed' for x in entries):
        return 'failed'
    if len(entries) == 9 and all(x['state'] == 'complete' for x in entries):
        return 'complete'
    return 'waiting'


def launch(task, side, seed):
    source = EXP / 'p1b_source' / task
    runner = source / 'reproduction' / ('run_roberta_cola_rdrop.py' if task == 'cola' else 'run_roberta_small_glue_regularization.py')
    ref = MAIN / 'references' / task / f'seed{seed}'
    out = RUNS / task / side / f'seed{seed}'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Refusing nonempty output: {out}')
    for path in (runner, ref / 'training_rng.pt', ref / 'reused_F/allocation.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    lr, epochs, wd, dropout = {'mrpc': ('0.00035','57','0.307','0.05'),
                                'cola': ('0.00019','42','0.196','0.0')}[task]
    cmd = [sys.executable, '-u', str(runner), '--task', task, '--seed', str(seed),
           '--output-dir', str(out), '--learning-rate', lr, '--epochs', epochs,
           '--scheduler-epochs', epochs, '--weight-decay', wd, '--lora-dropout', dropout,
           '--lr-scheduler', 'linear', '--calibration-steps', '16', '--rdrop-alpha', '1.0',
           '--subspace-consistency-lambda', '1.5', '--ablation-variant', 'F',
           '--allocation-reference', str(ref / 'reused_F/allocation.json'),
           '--training-rng-reference', str(ref / 'training_rng.pt')]
    if task == 'mrpc':
        cmd += ['--label-smoothing', '0.05']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', PYTHONHASHSEED=str(seed),
               SPECCON_SCORE_SIDE=side)
    log = LOGS / f'p1b_gpu3_{task}_{side}_seed{seed}.log'
    with log.open('ab') as stream:
        proc = subprocess.Popen(cmd, cwd=source, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    note(f'START task={task} side={side} seed={seed} gpu=3 pid={proc.pid}')
    return proc, out, log


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        entries = json.loads(STATUS.read_text())['entries']
        if [(x['task'], x['side'], x['seed']) for x in entries] != JOBS:
            raise RuntimeError('Existing queue differs from preregistration')
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
        entries = [{'task': task, 'side': side, 'seed': seed, 'state': 'queued',
                    'output': str(RUNS / task / side / f'seed{seed}')}
                   for task, side, seed in JOBS]
    note(f'START controller pid={os.getpid()} jobs={len(JOBS)} gpu=3')
    active = None
    failed = False
    while True:
        gate = gate_state()
        if gate == 'failed':
            save(entries, 'blocked_by_p1a_failure')
            note('STOP P1-A failed; P1-B not started')
            return 2
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
                note(f'EXIT {entry["task"]} side={entry["side"]} seed={entry["seed"]} state={entry["state"]} code={code}')
        if gate == 'complete' and active is None and not failed:
            queued = next((x for x in entries if x['state'] == 'queued'), None)
            if queued and gpu_admitted():
                try:
                    proc, out, log = launch(queued['task'], queued['side'], queued['seed'])
                    queued.update(state='running', gpu=3, pid=proc.pid, log=str(log))
                    active = (queued, proc, out)
                except Exception as exc:
                    queued.update(state='failed', error=str(exc))
                    failed = True
                    note(f'LAUNCH FAILED: {exc}')
        stage = 'waiting_for_p1a' if gate != 'complete' else 'running'
        if failed:
            stage = 'failed'
        elif active is None and all(x['state'] == 'complete' for x in entries):
            stage = 'complete'
        save(entries, stage)
        if active is None and (failed or stage == 'complete'):
            note('STOP failed=' + str(failed))
            return 1 if failed else 0
        time.sleep(30)


if __name__ == '__main__':
    sys.exit(main())
