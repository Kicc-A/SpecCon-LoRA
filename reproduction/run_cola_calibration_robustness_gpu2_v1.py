#!/usr/bin/env python3
"""Complete the frozen CoLA 5 x 3 x 2 calibration/training matrix on GPU 2."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('/workspace/SpecCon-LoRA')
OLD = PROJECT / 'experiments/extended_ablation_20261021_v2'
ROOT = PROJECT / 'experiments/cola_calibration_robustness_gpu2_v1'
MECH = PROJECT / 'experiments/independent_calibration_mechanism_20261019_v1'
GPU = 2
SEEDS = (777, 2026)
LAMBDAS = (0.0, 1.5)
HEAD_SHA = '8525992b7c0f62ea5e7d4d9ba33a20aa71acf73016426ef8315c03aa6386e112'


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def replace_arg(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def historical():
    old = read(OLD / 'manifest.json')
    rows = [j for j in old['jobs'] if j['spec']['group'] == 'calibration_subset'
            and j['spec']['task'] == 'cola' and j['spec']['seed'] == 42]
    assert len(rows) == 10
    rows.sort(key=lambda j: (j['spec']['replicate'], j['spec']['lambda']))
    for rep in range(5):
        pair = rows[2*rep:2*rep+2]
        assert [j['spec']['lambda'] for j in pair] == list(LAMBDAS)
        ids = [read(Path(j['output']) / 'calibration_rows.json') for j in pair]
        rng = [read(Path(j['output']) / 'training_rng.json') for j in pair]
        initial = [read(Path(j['output']) / 'initial_state.json') for j in pair]
        for j in pair:
            m = read(Path(j['output']) / 'metrics.json')
            hp = m['hyperparameters']
            assert m['state'] == 'complete' and m['epochs'] == 42 and m['final'] == m['selected']
            assert m['adapter_parameters'] == 755712 and m['seed'] == 42
            assert hp['lr'] == .00019 and hp['wd'] == .196
            assert hp['dropout'] == 0.0
            assert hp['rdrop_alpha'] == 1.0 and hp['training_forwards_per_step'] == 2
        assert ids[0]['ids'] == ids[1]['ids'] and ids[0]['blocks'] == ids[1]['blocks'] == 4
        assert len(ids[0]['ids']) == 512
        assert all(rng[0][k] == rng[1][k] for k in ('cpu_sha256', 'cuda_sha256'))
        assert initial[0]['classifier_sha256'] == initial[1]['classifier_sha256'] == HEAD_SHA
    return rows


def prepare():
    if (ROOT / 'manifest.json').exists():
        raise RuntimeError('Existing manifest; use run or status')
    original = historical()
    ROOT.mkdir(parents=True, exist_ok=True)
    old_source = OLD / 'source_snapshot/cola/seed42'
    frozen = ROOT / 'source_snapshot'
    shutil.copytree(old_source, frozen)
    runner = frozen / 'reproduction/run_roberta_cola_rdrop.py'
    source = runner.read_text()
    anchor = '    seed_everything(args.seed)\n'
    assert source.count(anchor) == 1
    # The cached gradients were measured from the seed42 classifier.  Keep the
    # model/head at seed42, while --seed still controls training data order and
    # the per-seed training RNG reference controls dropout.
    runner.write_text(source.replace(anchor,
        '    seed_everything(42)  # Model/head seed of frozen A-gradient cache.\n'))
    support = frozen / 'extended_ablation_support.py'
    text = support.read_text()
    needle = '    calibration = importlib.import_module("speccon_lora.calibration")\n'
    assert text.count(needle) == 1
    text = text.replace(needle,
        '    if spec.get("expected_classifier_sha256"):\n'
        '        assert classifier_before == spec["expected_classifier_sha256"], "Gradient cache/model head mismatch"\n'
        + needle)
    support.write_text(text)
    jobs = []
    checks = []
    for base in original:
        prior = Path(base['output'])
        old_initial = read(prior / 'initial_state.json')
        old_ids = read(prior / 'calibration_rows.json')
        old_rng = read(prior / 'training_rng.json')
        old_alloc = read(prior / 'allocation.json')
        assert old_initial['classifier_sha256'] == HEAD_SHA
        assert old_ids['ids'] == base['spec']['independent_ids']
        for seed in SEEDS:
            job = copy.deepcopy(base)
            ident = base['id'].replace('_s42', f'_s{seed}')
            spec = job['spec']
            spec['seed'] = seed
            spec['model_initialization_seed'] = 42
            spec['expected_classifier_sha256'] = HEAD_SHA
            spec['expected_initial_adapter_sha256'] = old_initial['adapter_sha256']
            job['id'] = ident
            job['cwd'] = str(frozen)
            job['output'] = str(ROOT / 'runs' / ident)
            job['log'] = str(ROOT / 'logs' / f'{ident}.log')
            job['reference'] = str(OLD / f'references/cola/seed{seed}')
            job['command'][2] = str(runner)
            for flag, value in (
                ('--seed', seed), ('--output-dir', job['output']),
                ('--extension-spec', ROOT / 'specs' / f'{ident}.json'),
                ('--allocation-reference', OLD / 'references/cola/seed42/allocation.json'),
                ('--training-rng-reference', Path(job['reference']) / 'training_rng.pt'),
            ):
                replace_arg(job['command'], flag, value)
            write(ROOT / 'specs' / f'{ident}.json', spec)
            reference_rng = read(Path(job['reference']) / 'training_rng.json')
            assert reference_rng['cpu_sha256'] != old_rng['cpu_sha256']
            assert old_alloc['rank_report']['budget_used'] == 755712
            checks.append({'id': ident, 'calibration_set': spec['replicate'], 'training_seed': seed,
                'lambda': spec['lambda'],
                'calibration_ids_sha256': hashlib.sha256(json.dumps(old_ids['ids']).encode()).hexdigest(),
                'raw_gradient': spec['independent_raw'],
                'cached_model_head_sha256': HEAD_SHA,
                'expected_initial_adapter_sha256': old_initial['adapter_sha256'],
                'training_cpu_rng_sha256': reference_rng['cpu_sha256'],
                'training_cuda_rng_sha256': reference_rng['cuda_sha256'],
                'budget': 755712, 'source_job': base['id']})
            jobs.append(job)
    assert len(jobs) == 20 and len({j['id'] for j in jobs}) == 20
    jobs.sort(key=lambda j: (j['spec']['replicate'], j['spec']['seed'], j['spec']['lambda']))
    checks.sort(key=lambda j: (j['calibration_set'], j['training_seed'], j['lambda']))
    for rep in range(5):
        for seed in SEEDS:
            pair = [r for r in checks if r['calibration_set'] == rep and r['training_seed'] == seed]
            assert len(pair) == 2
            for field in ('calibration_ids_sha256', 'raw_gradient', 'cached_model_head_sha256',
                          'training_cpu_rng_sha256', 'training_cuda_rng_sha256', 'budget'):
                assert pair[0][field] == pair[1][field], (rep, seed, field)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in frozen.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    write(ROOT / 'pretraining_pair_checks.json', {'created': now(), 'passed': True,
        'scope': '20 planned new runs; pair checks are static before training',
        'model_initialization_seed': 42, 'rows': checks})
    write(ROOT / 'manifest.json', {'created': now(), 'gpu': GPU,
        'scope': 'EXP-01 CoLA 5 calibration sets x seeds 42,777,2026 x two lambda arms',
        'model_initialization_seed': 42, 'training_seeds': [42, *SEEDS],
        'reuse': [j['id'] for j in original],
        'old_manifest': str(OLD / 'manifest.json'),
        'old_manifest_sha256': sha(OLD / 'manifest.json'),
        'source_hashes': hashes, 'jobs': jobs})
    write(ROOT / 'status.json', {'updated': now(), 'state': 'prepared', 'total_new': 20,
        'complete_new': 0, 'active': None})
    (ROOT / 'PROTOCOL.md').write_text(
        '# CoLA calibration robustness completion\n\n'
        'Reuses 10 frozen seed42 runs; adds 20 runs for training seeds 777 and 2026. '
        'The A-gradient caches were collected with model and classifier seed42, so all 30 runs '
        'use the identical seed42 model/head. Training seed controls epoch-wise RandomSampler '
        'and the per-seed restored CPU/CUDA RNG immediately before training. Thus this design '
        'isolates training stochasticity under a common initial model. It does not test new '
        'classifier initializations or fresh gradient caches. The original frozen CoLA source '
        'is copied; only the model-seed line and explicit head hash assertion are added. '
        'Each new arm asserts the exact original initial adapter hash before optimization. '
        'GPU2 runs one new task at a time while its free-memory gate is satisfied. '
        'All training uses the fixed final checkpoint and frozen optimizer settings.\n')
    print(now(), 'PREPARED', len(jobs), 'new jobs; pretraining pair checks passed', flush=True)


def free_memory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    return {int(line.split(',')[0]): int(line.split(',')[1]) for line in raw.splitlines()}[GPU]


def running_pid(job):
    expected = job['output'].encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'--output-dir' in args and args[args.index(b'--output-dir') + 1] == expected:
                return int(proc.name)
        except (OSError, ValueError, IndexError):
            pass
    return None


def validate(job):
    sys.path.insert(0, str(PROJECT / 'reproduction'))
    from launch_extended_ablation_20261020 import validate as historical_validate
    row = historical_validate(job)
    state = read(Path(job['output']) / 'initial_state.json')
    assert state['classifier_sha256'] == HEAD_SHA
    assert state['adapter_sha256'] == job['spec']['expected_initial_adapter_sha256']
    assert read(Path(job['output']) / 'calibration_rows.json')['ids'] == job['spec']['independent_ids']
    return row


def launch(job):
    env = os.environ.copy()
    for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES=str(GPU), PYTHONPATH=job['cwd'], OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        PYTHONHASHSEED=str(job['spec']['seed']), HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_DISABLED='true')
    Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
    with Path(job['log']).open('a') as log:
        proc = subprocess.Popen(job['command'], cwd=job['cwd'], env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    return proc.pid


def run():
    with (ROOT / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(ROOT / 'manifest.json')
        assert sha(OLD / 'manifest.json') == manifest['old_manifest_sha256']
        for relative, digest in manifest['source_hashes'].items():
            assert sha(ROOT / relative) == digest, relative
        jobs = manifest['jobs']
        completed = read(ROOT / 'completed_new.json') if (ROOT / 'completed_new.json').exists() else {}
        while len(completed) < len(jobs):
            if (ROOT / 'STOP').exists():
                write(ROOT / 'status.json', {'updated': now(), 'state': 'stopped',
                    'total_new': len(jobs), 'complete_new': len(completed), 'active': None})
                return
            job = next(j for j in jobs if j['id'] not in completed)
            out = Path(job['output'])
            if (out / 'metrics.json').exists():
                completed[job['id']] = validate(job)
                write(ROOT / 'completed_new.json', completed)
                continue
            pid = running_pid(job)
            if pid is None:
                if out.exists() and any(out.iterdir()):
                    archive = ROOT / 'attempts' / f'{job["id"]}_{int(time.time())}'
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    out.rename(archive)
                free = free_memory()
                required = job['min_free_mib'] + 1024
                if free < required:
                    write(ROOT / 'status.json', {'updated': now(), 'state': 'waiting_for_memory',
                        'total_new': len(jobs), 'complete_new': len(completed),
                        'next': job['id'], 'free_mib': free, 'required_mib': required, 'active': None})
                    time.sleep(30)
                    continue
                pid = launch(job)
                print(now(), 'START', job['id'], 'gpu', GPU, 'pid', pid, flush=True)
            write(ROOT / 'status.json', {'updated': now(), 'state': 'running',
                'total_new': len(jobs), 'complete_new': len(completed),
                'active': {'id': job['id'], 'pid': pid, 'gpu': GPU, 'output': job['output']}})
            while running_pid(job) == pid:
                time.sleep(30)
            if not (out / 'metrics.json').exists():
                raise RuntimeError(f'Training ended before metrics.json: {job["log"]}')
            completed[job['id']] = validate(job)
            write(ROOT / 'completed_new.json', completed)
            print(now(), 'DONE', job['id'], completed[job['id']]['score'], flush=True)
            if len(completed) == 2:
                first = [completed[j['id']] for j in jobs[:2]]
                assert first[0]['seed'] == first[1]['seed'] == 777
                assert first[0]['replicate'] == first[1]['replicate'] == 0
                for field in ('ids', 'blocks'):
                    x = read(Path(jobs[0]['output']) / 'calibration_rows.json')[field]
                    y = read(Path(jobs[1]['output']) / 'calibration_rows.json')[field]
                    assert x == y
                write(ROOT / 'first_pair_validation.json', {'updated': now(),
                    'passed': True, 'training_seed': 777, 'calibration_set': 0,
                'energy_score': first[0]['score'], 'consensus_score': first[1]['score'],
                    'delta': first[1]['score'] - first[0]['score']})
        write(ROOT / 'status.json', {'updated': now(), 'state': 'complete',
            'total_new': len(jobs), 'complete_new': len(completed), 'active': None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'status'))
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'run':
        run()
    else:
        print(json.dumps(read(ROOT / 'status.json'), ensure_ascii=False, indent=2))
