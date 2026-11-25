#!/usr/bin/env python3
"""Queue the preregistered MRPC/CoLA shuffled-consensus pilot."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXP = Path('/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1')
MAIN = Path('/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3')
SOURCE = EXP / 'p1a_source'
RUNS = EXP / 'runs/p1a_shuffle_seed20261106'
LOGS = EXP / 'logs'
STATUS = EXP / 'status_p1a_shuffle.json'
CONTROL_LOG = EXP / 'controller_p1a_shuffle.log'
JOBS = [(task, seed) for task in ('mrpc', 'cola') for seed in (42, 777, 2026)]


def note(message):
    line = time.strftime('%Y-%m-%dT%H:%M:%S%z') + ' ' + message
    with CONTROL_LOG.open('a') as stream:
        stream.write(line + '\n')
    print(line, flush=True)


def save(entries):
    payload = {'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'controller_pid': os.getpid(), 'entries': entries}
    tmp = STATUS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(STATUS)


def gpu_rows():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    return [tuple(map(int, line.split(','))) for line in out.splitlines()]


def eligible(busy):
    candidates = []
    for gpu, free, _ in gpu_rows():
        if gpu in busy or free < 9 * 1024:
            continue
        readings = []
        for _ in range(3):
            readings.append(next(util for index, _, util in gpu_rows() if index == gpu))
            time.sleep(2)
        if sum(readings) / len(readings) <= 75:
            candidates.append((gpu, free))
    return [gpu for gpu, _ in sorted(candidates, key=lambda item: -item[1])]


def launch(task, seed, gpu):
    out = RUNS / task / f'seed{seed}'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Nonempty output already exists: {out}')
    source = SOURCE / task
    runner = source / 'reproduction' / ('run_roberta_cola_rdrop.py' if task == 'cola' else 'run_roberta_small_glue_regularization.py')
    reference = MAIN / 'references' / task / f'seed{seed}'
    rng = reference / 'training_rng.pt'
    allocation = reference / 'reused_F' / 'allocation.json'
    for path in (runner, rng, allocation):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = {'mrpc': {'lr': '0.00035', 'epochs': '57', 'wd': '0.307', 'dropout': '0.05'},
              'cola': {'lr': '0.00019', 'epochs': '42', 'wd': '0.196', 'dropout': '0.0'}}[task]
    cmd = [sys.executable, '-u', str(runner), '--task', task, '--seed', str(seed),
           '--output-dir', str(out), '--learning-rate', config['lr'],
           '--epochs', config['epochs'], '--scheduler-epochs', config['epochs'],
           '--weight-decay', config['wd'], '--lora-dropout', config['dropout'],
           '--lr-scheduler', 'linear', '--calibration-steps', '16', '--rdrop-alpha', '1.0',
           '--subspace-consistency-lambda', '1.5', '--ablation-variant', 'F',
           '--allocation-reference', str(allocation), '--training-rng-reference', str(rng)]
    if task == 'mrpc':
        cmd += ['--label-smoothing', '0.05']
    manifest = out / 'shuffle_manifest.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(seed),
               SPECCON_SHUFFLE_SEED='20261106', SPECCON_SHUFFLE_MANIFEST=str(manifest))
    log = LOGS / f'p1a_shuffle_{task}_seed{seed}.log'
    with log.open('ab') as stream:
        proc = subprocess.Popen(cmd, cwd=source, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    note(f'START {task} seed={seed} gpu={gpu} pid={proc.pid} output={out}')
    return proc, out, log


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        entries = json.loads(STATUS.read_text())['entries']
        if [(e['task'], e['seed']) for e in entries] != JOBS:
            raise RuntimeError('Existing queue does not match preregistered jobs')
        for entry in entries:
            out = Path(entry['output'])
            metrics_path = out / 'metrics.json'
            try:
                metrics = json.loads(metrics_path.read_text())
            except (OSError, json.JSONDecodeError):
                metrics = {}
            if metrics.get('state') == 'complete' and metrics.get('adapter_parameters') == 755712:
                entry['state'] = 'complete'
            elif entry['state'] == 'running':
                raise RuntimeError(f'Stale running job lacks valid complete metrics: {out}')
    else:
        entries = [{'task': task, 'seed': seed, 'state': 'queued',
                    'output': str(RUNS / task / f'seed{seed}')} for task, seed in JOBS]
    active = {}
    note(f'START controller pid={os.getpid()} jobs={len(JOBS)}')
    save(entries)
    failed = False
    while True:
        for key, (proc, out, _) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[key]
            entry = next(e for e in entries if (e['task'], e['seed']) == key)
            try:
                metrics = json.loads((out / 'metrics.json').read_text())
            except (OSError, json.JSONDecodeError):
                metrics = {}
            ok = code == 0 and metrics.get('state') == 'complete' and metrics.get('adapter_parameters') == 755712
            entry['state'] = 'complete' if ok else 'failed'
            entry['exit_code'] = code
            failed |= not ok
            note(f'EXIT {key} state={entry["state"]} code={code}')
        if not failed:
            for entry in entries:
                if entry['state'] != 'queued' or len(active) >= 2:
                    continue
                gpus = eligible({e['gpu'] for e in entries if (e['task'], e['seed']) in active})
                if not gpus:
                    break
                key = (entry['task'], entry['seed'])
                try:
                    proc, out, log = launch(*key, gpus[0])
                except Exception as exc:
                    entry.update(state='failed', error=str(exc))
                    failed = True
                    note(f'LAUNCH FAILED {key}: {exc}')
                    break
                active[key] = (proc, out, log)
                entry.update(state='running', gpu=gpus[0], pid=proc.pid, log=str(log))
        save(entries)
        if not active and (failed or all(e['state'] == 'complete' for e in entries)):
            note('STOP failed=' + str(failed))
            return 1 if failed else 0
        time.sleep(30)


if __name__ == '__main__':
    sys.exit(main())
