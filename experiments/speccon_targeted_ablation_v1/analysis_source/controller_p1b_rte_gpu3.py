#!/usr/bin/env python3
"""RTE right/joint consensus controls, one process at a time on GPU 3."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXP = Path('/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1')
MAIN = Path('/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3')
SOURCE = EXP / 'p1b_source/rte'
RUNNER = SOURCE / 'reproduction/run_roberta_small_glue_rdrop.py'
RUNS = EXP / 'runs/p1b_rte_gpu3_score_side_20261107'
LOGS = EXP / 'logs'
STATUS = EXP / 'status_p1b_rte_gpu3.json'
CONTROL_LOG = EXP / 'controller_p1b_rte_gpu3.log'
JOBS = [(side, seed) for side in ('right', 'joint') for seed in (42, 777, 2026)]


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
        raw = subprocess.check_output(['nvidia-smi', '-i', '3',
                                       '--query-gpu=memory.free,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        free, util = (int(x) for x in raw.strip().split(','))
        samples.append((free, util))
        time.sleep(2)
    # Previous RTE runs peaked at 10.93 GiB allocated; leave >3 GiB headroom.
    return min(x[0] for x in samples) >= 14 * 1024 and sum(x[1] for x in samples) / 3 <= 75


def launch(side, seed):
    out = RUNS / side / f'seed{seed}'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Refusing nonempty output: {out}')
    ref = MAIN / 'references/rte' / f'seed{seed}'
    for p in (RUNNER, ref / 'training_rng.pt', ref / 'reused_F/allocation.json'):
        if not p.is_file():
            raise FileNotFoundError(p)
    command = [sys.executable, '-u', str(RUNNER), '--task', 'rte', '--seed', str(seed),
               '--output-dir', str(out), '--learning-rate', '0.00021', '--epochs', '63',
               '--scheduler-epochs', '63', '--weight-decay', '0.198', '--lora-dropout', '0.0',
               '--lr-scheduler', 'linear', '--calibration-steps', '16', '--rdrop-alpha', '1.0',
               '--rank-score-gamma', '1.2', '--subspace-consistency-lambda', '0.5',
               '--ablation-variant', 'F',
               '--allocation-reference', str(ref / 'reused_F/allocation.json'),
               '--training-rng-reference', str(ref / 'training_rng.pt')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', PYTHONHASHSEED=str(seed),
               SPECCON_SCORE_SIDE=side)
    log = LOGS / f'p1b_rte_gpu3_{side}_seed{seed}.log'
    with log.open('ab') as stream:
        proc = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    note(f'START side={side} seed={seed} gpu=3 pid={proc.pid}')
    return proc, out, log


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        entries = json.loads(STATUS.read_text())['entries']
        if [(x['side'], x['seed']) for x in entries] != JOBS:
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
        entries = [{'side': side, 'seed': seed, 'state': 'queued',
                    'output': str(RUNS / side / f'seed{seed}')}
                   for side, seed in JOBS]
    note(f'START controller pid={os.getpid()} jobs={len(JOBS)} gpu=3')
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
                note(f'EXIT side={entry["side"]} seed={entry["seed"]} state={entry["state"]} code={code}')
        if active is None and not failed:
            queued = next((x for x in entries if x['state'] == 'queued'), None)
            if queued and gpu_admitted():
                try:
                    proc, out, log = launch(queued['side'], queued['seed'])
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
