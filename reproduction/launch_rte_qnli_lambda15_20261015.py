#!/usr/bin/env python3
"""Replay retained RTE/QNLI settings with lambda=1.5 and three paired seeds."""
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
ROOT = PROJECT / 'experiments/rte_qnli_lambda15_3seeds_gpu012_20261015_v1'
REPORT = PROJECT / 'reports/glue_best_42_777_2026_20261013/results.json'
SEEDS = [42, 777, 2026]
RUNNERS = {'rte': 'run_roberta_small_glue_rdrop.py',
           'qnli': 'run_roberta_other_glue_rdrop.py'}
# Historical allocated peaks plus allocator/context headroom, in MiB.
RESERVATION_MIB = {'rte': 12500, 'qnli': 10000}
HEADROOM_MIB = 1024


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


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(index): int(free) for index, free in
            (line.split(',') for line in output.strip().splitlines())}


def prepare():
    report = read(REPORT)
    assert report['seeds'] == SEEDS
    records = []
    for task in RUNNERS:
        for row in report['tasks'][task]['runs']:
            source = PROJECT / row['source']
            config = read(source / 'run_config.json')
            snapshot = next(p / 'source_snapshot' for p in source.parents
                            if (p / 'source_snapshot').is_dir())
            runner = snapshot / 'reproduction' / RUNNERS[task]
            assert digest(runner) == config['source_sha256']
            assert Path(config['config']['training_rng_reference']).is_file()
            assert config['task'] == task and config['seed'] == row['seed']
            records.append((task, row, source, config, snapshot))
    assert len(records) == 6
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'logs').mkdir()
    jobs = []
    for task, row, source, config, snapshot in records:
        target = ROOT / 'source_snapshot' / task
        if not target.exists():
            shutil.copytree(snapshot / 'speccon_lora', target / 'speccon_lora',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            (target / 'reproduction').mkdir()
            shutil.copy2(snapshot / 'reproduction' / RUNNERS[task],
                         target / 'reproduction' / RUNNERS[task])
        seed = row['seed']
        reference = ROOT / 'references' / task / f'seed{seed}'
        reference.mkdir(parents=True)
        for filename in ('run_config.json', 'metrics.json', 'training_rng.json'):
            if (source / filename).exists():
                shutil.copy2(source / filename, reference / filename)
        rng = Path(config['config']['training_rng_reference'])
        shutil.copy2(rng, reference / 'training_rng.pt')
        output = ROOT / 'runs' / task / f'seed{seed}'
        hp = config['config']
        command = [sys.executable, '-u', str(target / 'reproduction' / RUNNERS[task]),
                   '--task', task, '--seed', str(seed), '--output-dir', str(output),
                   '--epochs', str(hp['epochs']), '--scheduler-epochs', str(hp['scheduler_epochs']),
                   '--learning-rate', str(hp['lr']), '--weight-decay', str(hp['wd']),
                   '--lora-dropout', str(config['lora_dropout']),
                   '--lr-scheduler', hp['lr_scheduler'],
                   '--calibration-steps', str(hp['calibration_steps']),
                   '--rdrop-alpha', str(hp['rdrop_alpha']),
                   '--subspace-consistency-lambda', '1.5',
                   '--training-rng-reference', str(rng)]
        if task == 'rte':
            command += ['--rank-score-gamma', str(config['rank_score_gamma'])]
        expected = dict(config, subspace_consistency_lambda=1.5)
        jobs.append(dict(task=task, seed=seed, command=command, cwd=str(target),
                         output=str(output), log=str(ROOT / 'logs' / f'{task}_seed{seed}.log'),
                         baseline_source=str(source), baseline_score=row['score'],
                         expected_run_config=expected,
                         rng_reference=str(rng), rng_sha256=digest(rng)))
    shutil.copy2(REPORT, ROOT / 'references' / 'previous_report.json')
    shutil.copy2(__file__, ROOT / 'controller.py')
    hashes = {str(p.relative_to(ROOT)): digest(p)
              for p in (ROOT / 'source_snapshot').rglob('*.py')}
    hashes['controller.py'] = digest(ROOT / 'controller.py')
    write(ROOT / 'manifest.json', dict(created=now(), seeds=SEEDS, tasks=list(RUNNERS),
          change={'rte': {'old_lambda': 0.5, 'lambda': 1.5},
                  'qnli': {'old_lambda': 0.0, 'lambda': 1.5}},
          scheduling=None, jobs=jobs, source_sha256=hashes,
          protocol='All training settings and paired RNG states retained; only lambda changes.'))
    write(ROOT / 'status.json', dict(state='prepared', updated=now(), total=6,
          completed=[], active=[], pending=[{'task': j['task'], 'seed': j['seed']} for j in jobs], failed=[]))
    print(json.dumps({'prepared': str(ROOT), 'runs': len(jobs)}))


def validate(info):
    output = Path(info['output'])
    assert read(output / 'run_config.json') == info['expected_run_config']
    metric = read(output / 'metrics.json')
    expected_hp = read(Path(info['baseline_source']) / 'metrics.json')['hyperparameters']
    assert metric['hyperparameters'] == dict(expected_hp, **{'lambda': 1.5})
    assert metric['state'] == 'complete' and metric['task'] == info['task']
    assert metric['seed'] == info['seed']
    assert metric['epochs'] == info['expected_run_config']['config']['epochs']
    assert metric['selected'] == metric['final'] and metric['final']['epoch'] == metric['epochs']
    assert metric['precision'] == 'FP32' and metric['global_batch_size'] == 32
    assert metric['adapter_parameters'] == 755712
    assert len((output / 'history.jsonl').read_text().splitlines()) == metric['epochs']
    return dict(task=info['task'], seed=info['seed'], gpu=info['gpu'], output=str(output),
                score=metric['final_score_percent'], baseline_score=info['baseline_score'],
                delta=metric['final_score_percent']-info['baseline_score'], finished=now())


def publish(active, pending, completed, failed):
    active_rows = []
    for process, info in active:
        status = Path(info['output']) / 'status.json'
        active_rows.append(dict(task=info['task'], seed=info['seed'], gpu=info['gpu'],
                               pid=process.pid, log=info['log'], output=info['output'],
                               training_status=read(status) if status.exists() else {}))
    state = ('failed_draining' if active else 'failed') if failed else (
        'running' if active or pending else 'complete')
    write(ROOT / 'status.json', dict(state=state, updated=now(), controller_pid=os.getpid(),
          total=6, completed=completed, active=active_rows,
          pending=[{'task': j['task'], 'seed': j['seed'], 'gpu': j['gpu']} for j in pending], failed=failed))
    groups = {}
    for task in RUNNERS:
        rows = [r for r in completed if r['task'] == task]
        if rows:
            groups[task] = dict(completed=len(rows), mean=statistics.mean(r['score'] for r in rows),
                                sample_sd=statistics.stdev(r['score'] for r in rows) if len(rows)>1 else None)
    write(ROOT / 'results.json', dict(updated=now(), runs=completed, groups=groups))


def controller():
    manifest = read(ROOT / 'manifest.json')
    for relative, checksum in manifest['source_sha256'].items():
        assert digest(ROOT / relative) == checksum, relative
    pending = list(manifest['jobs'])
    initial_free = {int(gpu): free for gpu, free in manifest['initial_free_mib'].items()}
    active, completed, failed = [], [], []
    try:
        while pending or active:
            remaining = []
            for process, info in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, info))
                    continue
                try:
                    if code:
                        raise RuntimeError(f'exit {code}; see {info["log"]}')
                    completed.append(validate(info))
                except Exception as error:
                    failed.append(dict(task=info['task'], seed=info['seed'], error=repr(error), log=info['log']))
                print(json.dumps({'finished': info['task'], 'seed': info['seed'], 'exit_code': code}), flush=True)
            active = remaining
            if not failed:
                free_now = gpu_memory()
                for job in list(pending):
                    gpu = job['gpu']
                    colocated = [info for _, info in active if info['gpu'] == gpu]
                    if manifest['scheduling'] == 'seed-per-gpu' and colocated:
                        continue
                    reserved = sum(RESERVATION_MIB[info['task']] for info in colocated)
                    available = min(free_now[gpu], initial_free[gpu] - reserved) - HEADROOM_MIB
                    if available < RESERVATION_MIB[job['task']]:
                        continue
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']),
                        PYTHONHASHSEED=str(job['seed']), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                        HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                    assert digest(job['rng_reference']) == job['rng_sha256']
                    assert not Path(job['output']).exists()
                    with Path(job['log']).open('x') as stream:
                        process = subprocess.Popen(job['command'], cwd=job['cwd'], env=env,
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    active.append((process, job))
                    pending.remove(job)
                    print(json.dumps({'started': job['task'], 'seed': job['seed'],
                                      'gpu': job['gpu'], 'pid': process.pid}), flush=True)
            publish(active, pending, completed, failed)
            if failed and not active:
                return 1
            if active or pending:
                time.sleep(10)
    except Exception as error:
        write(ROOT / 'controller_error.json', dict(error=repr(error), updated=now(),
              active=[{'task': info['task'], 'seed': info['seed'], 'pid': process.pid}
                      for process, info in active]))
        raise
    return 0


def launch(mode):
    assert read(ROOT / 'status.json')['state'] == 'prepared'
    manifest = read(ROOT / 'manifest.json')
    manifest['scheduling'] = mode
    manifest['initial_free_mib'] = gpu_memory()
    manifest['reservation_mib'] = RESERVATION_MIB
    manifest['headroom_mib'] = HEADROOM_MIB
    for job in manifest['jobs']:
        job['gpu'] = SEEDS.index(job['seed']) if mode == 'seed-per-gpu' else list(RUNNERS).index(job['task'])
    write(ROOT / 'manifest.json', manifest)
    with (ROOT / 'controller.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'controller.py'), '--controller'],
            cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(ROOT / 'controller_pid.json', dict(pid=process.pid, started=now()))
    print(json.dumps(dict(pid=process.pid, root=str(ROOT), scheduling=mode)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--controller', action='store_true')
    group.add_argument('--launch', choices=['seed-per-gpu', 'dataset-per-gpu'])
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.controller:
        sys.exit(controller())
    else:
        launch(args.launch)
