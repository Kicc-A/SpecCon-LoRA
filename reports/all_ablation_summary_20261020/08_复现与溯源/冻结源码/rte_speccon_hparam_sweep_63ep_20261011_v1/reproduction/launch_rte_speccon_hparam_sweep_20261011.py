#!/usr/bin/env python3
"""Three-seed, fixed-final RTE sweep at the original 63-epoch budget."""
import argparse
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
ROOT = PROJECT / 'experiments/rte_speccon_hparam_sweep_63ep_20261011_v1'
BASE = PROJECT / 'experiments/glue_rdrop_best_seed2026_gpu01_20261008_v1'
SEEDS = [42, 777, 2026]
BASELINE = dict(lr=2.1e-4, gamma=1.2, lam=1.5)
# One factor at a time, with the published parameter budget and training setup fixed.
CONFIGS = [
    dict(name='lr18', lr=1.8e-4, gamma=1.2, lam=1.5),
    dict(name='lr25', lr=2.5e-4, gamma=1.2, lam=1.5),
    dict(name='lr32', lr=3.2e-4, gamma=1.2, lam=1.5),
    dict(name='gamma07', lr=2.1e-4, gamma=0.7, lam=1.5),
    dict(name='gamma16', lr=2.1e-4, gamma=1.6, lam=1.5),
    dict(name='lambda00', lr=2.1e-4, gamma=1.2, lam=0.0),
    dict(name='lambda05', lr=2.1e-4, gamma=1.2, lam=0.5),
    dict(name='lambda30', lr=2.1e-4, gamma=1.2, lam=3.0),
]


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


def jobs():
    return [dict(config=config, seed=seed) for config in CONFIGS for seed in SEEDS]


def prepare():
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'logs').mkdir()
    snapshot = ROOT / 'source_snapshot'
    shutil.copytree(BASE / 'source_snapshot/speccon_lora', snapshot / 'speccon_lora',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (snapshot / 'reproduction').mkdir()
    for filename in ['run_roberta_small_glue_rdrop.py', Path(__file__).name]:
        shutil.copy2(PROJECT / 'reproduction' / filename, snapshot / 'reproduction' / filename)
    summary = read(BASE / 'summary.json')
    rte = next(group for group in summary['groups'] if group['task'] == 'rte')
    refs = {}
    for item in rte['per_seed']:
        metric = read(Path(item['source']) / 'metrics.json')
        assert metric['state'] == 'complete' and metric['epochs'] == 63
        assert metric['hyperparameters']['gamma'] == BASELINE['gamma']
        assert metric['hyperparameters']['lambda'] == BASELINE['lam']
        assert metric['hyperparameters']['lr'] == BASELINE['lr']
        refs[str(item['seed'])] = dict(score=item['score'], source=item['source'],
                                      rng=str(Path(item['source']) / 'training_rng.pt'))
    assert set(refs) == set(map(str, SEEDS))
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in snapshot.rglob('*.py')}
    write(ROOT / 'manifest.json', dict(created=now(), task='rte', model='RoBERTa-base',
          epochs=63, scheduler_epochs=63, seeds=SEEDS, baseline=BASELINE,
          configs=CONFIGS, jobs=jobs(), reference=refs,
          fixed=dict(weight_decay=0.198, lora_dropout=0.0, rdrop_alpha=1.0,
                     calibration_steps=16, parameter_budget=755712,
                     selection='fixed final epoch validation accuracy'),
          source_sha256=hashes))
    write(ROOT / 'status.json', dict(state='prepared', updated=now(), total=24,
                                     completed=0, active=[], pending=jobs(), failed=[]))


def gpu_info(index):
    output = subprocess.check_output([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    memory, utilization = [int(value.strip()) for value in output.strip().split(',')]
    return memory, utilization


def launch(job, gpu):
    cfg, seed = job['config'], job['seed']
    output = ROOT / 'runs' / cfg['name'] / f'seed{seed}'
    log = ROOT / 'logs' / f'{cfg["name"]}_seed{seed}.log'
    assert not output.exists() and not log.exists()
    command = [sys.executable, '-u', str(ROOT / 'source_snapshot/reproduction/run_roberta_small_glue_rdrop.py'),
               '--task', 'rte', '--seed', str(seed), '--output-dir', str(output),
               '--epochs', '63', '--scheduler-epochs', '63', '--learning-rate', str(cfg['lr']),
               '--weight-decay', '0.198', '--lora-dropout', '0.0', '--lr-scheduler', 'linear',
               '--calibration-steps', '16', '--rdrop-alpha', '1.0',
               '--rank-score-gamma', str(cfg['gamma']),
               '--subspace-consistency-lambda', str(cfg['lam']),
               '--training-rng-reference', read(ROOT / 'manifest.json')['reference'][str(seed)]['rng']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(seed),
               CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT / 'source_snapshot', env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    info = dict(job, gpu=gpu, pid=process.pid, output=str(output), log=str(log), started=now())
    return process, info


def validate(info):
    metric = read(Path(info['output']) / 'metrics.json')
    cfg, seed = info['config'], info['seed']
    hp = metric['hyperparameters']
    assert metric['state'] == 'complete' and metric['task'] == 'rte'
    assert metric['seed'] == seed and metric['epochs'] == 63
    assert metric['final']['epoch'] == 63 and metric['selected'] == metric['final']
    assert hp['scheduler_epochs'] == 63 and hp['rdrop_alpha'] == 1.0
    assert hp['lr'] == cfg['lr'] and hp['gamma'] == cfg['gamma'] and hp['lambda'] == cfg['lam']
    assert metric['adapter_parameters'] == 755712
    history = (Path(info['output']) / 'history.jsonl').read_text().splitlines()
    assert len(history) == 63
    score = 100 * metric['final']['splits']['validation']['accuracy']
    baseline = read(ROOT / 'manifest.json')['reference'][str(seed)]['score']
    return dict(config=cfg, seed=seed, score=score, baseline=baseline,
                delta=score-baseline, output=info['output'], gpu=info['gpu'])


def report(done, active, pending, failed):
    write(ROOT / 'results.json', dict(updated=now(), runs=done))
    write(ROOT / 'status.json', dict(updated=now(), total=24, completed=len(done),
          state='running' if active else 'failed' if failed else 'waiting_for_gpu' if pending else 'complete',
          active=[dict(info, training_status=read(Path(info['output']) / 'status.json')
                       if (Path(info['output']) / 'status.json').exists() else {})
                  for _, info in active], pending=pending, failed=failed))
    lines = ['# RTE SpecCon-LoRA 参数搜索', '',
             '固定 63 轮、最终轮验证集准确率；每次只改变学习率、gamma 或 lambda 之一。',
             '基线：LR=2.1e-4，gamma=1.2，lambda=1.5；三 seed 平均 79.18%。', '',
             '|配置|已完成 seed|准确率均值|对配对基线差值|', '|---|---:|---:|---:|']
    for cfg in CONFIGS:
        rows = [row for row in done if row['config']['name'] == cfg['name']]
        if rows:
            lines.append(f"|{cfg['name']}|{len(rows)}/3|{sum(x['score'] for x in rows)/len(rows):.2f}|{sum(x['delta'] for x in rows)/len(rows):+.2f}|")
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def controller():
    manifest = read(ROOT / 'manifest.json')
    for filename, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'source_snapshot' / filename).read_bytes()).hexdigest() == digest
    pending, active, done, failed = list(manifest['jobs']), [], [], []
    while pending or active:
        remaining = []
        for process, info in active:
            code = process.poll()
            if code is None:
                remaining.append((process, info))
                continue
            try:
                if code != 0:
                    raise RuntimeError(f'exit {code}; see {info["log"]}')
                done.append(validate(info))
            except Exception as error:
                failed.append(dict(info, error=repr(error)))
        active = remaining
        if not failed:
            for gpu in [1, 3]:
                if not pending or any(info['gpu'] == gpu for _, info in active):
                    continue
                free, utilization = gpu_info(gpu)
                if free >= 12500 and utilization <= 85:
                    active.append(launch(pending.pop(0), gpu))
        report(done, active, pending, failed)
        if failed and not active:
            return 1
        if pending or active:
            time.sleep(20)
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
