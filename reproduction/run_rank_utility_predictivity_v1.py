#!/usr/bin/env python3
"""Frozen MRPC near-uniform reference plus pre-registered rank swaps on GPU 1."""
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
EXP01 = PROJECT / 'experiments/mrpc_calibration_robustness_v1'
ROOT = PROJECT / 'experiments/rank_utility_predictivity_v1'
RAW = PROJECT / 'experiments/independent_calibration_mechanism_20261019_v1/runs/mrpc/a0_order0_raw.pt'
GPU = 1
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


def set_arg(command, flag, value):
    command[command.index(flag)+1] = str(value)


def prepare():
    if (ROOT / 'manifest.json').exists():
        raise RuntimeError('Existing EXP-02 manifest; use run or status')
    registered = read(ROOT / 'pre_registered_swaps.json')
    assert registered['pre_registered_before_training'] and len(registered['selected']) == 12
    allocations = read(ROOT / 'swap_rank_allocation.json')
    old = read(OLD / 'manifest.json')
    baseline = next(j for j in old['jobs'] if j['spec']['task']=='mrpc'
        and j['spec']['seed']==42 and j['spec']['name']=='near_uniform_svd')
    frozen = ROOT / 'source_snapshot'
    shutil.copytree(OLD / 'source_snapshot/mrpc/seed42', frozen)
    runner = frozen / 'reproduction/run_roberta_small_glue_regularization.py'
    source = runner.read_text()
    anchor = '    seed_everything(args.seed)\n'
    assert source.count(anchor) == 1
    runner.write_text(source.replace(anchor,
        '    seed_everything(42)  # A0 gradient cache and classifier use seed42.\n'))
    support = frozen / 'extended_ablation_support.py'
    text = support.read_text()
    anchor = '    calibration = importlib.import_module("speccon_lora.calibration")\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor,
        '    if spec.get("expected_classifier_sha256"):\n'
        '        assert classifier_before == spec["expected_classifier_sha256"], "Gradient cache/model head mismatch"\n'
        + anchor)
    anchor = '        if spec["allocation"] == "adaptive":\n'
    assert text.count(anchor) == 1
    fixed = (
        '        if spec.get("fixed_rank_map") is not None:\n'
        '            ranks = {n: int(r) for n, r in spec["fixed_rank_map"].items()}\n'
        '            assert set(ranks) == set(gradients)\n'
        '            assert all(cfg.r_min <= r <= cfg.r_max for r in ranks.values())\n'
        '            used = sum(ranks[n] * sum(v["grad"].shape) for n, v in gradients.items())\n'
        '            assert used == cfg.param_budget\n'
        '            stats = {n: {"rank": r, "allocation": "pre_registered_fixed_swap"} for n, r in ranks.items()}\n'
        '            stats["__summary__"] = dict(budget_target=used, budget_used=used,\n'
        '                total_adapter_params=used, budget_unused=0, rank_min=min(ranks.values()),\n'
        '                rank_max=max(ranks.values()), rank_mean=sum(ranks.values()) / len(ranks))\n'
        '        elif spec["allocation"] == "adaptive":\n'
    )
    text = text.replace(anchor, fixed)
    support.write_text(text)
    jobs = []
    for name, rank_map in [('reference', allocations['reference'])] + [
            (item['swap_id'], allocations['swaps'][item['swap_id']]) for item in registered['selected']]:
        job = copy.deepcopy(baseline)
        ident = f'mrpc_{name}_s42'
        spec = job['spec']
        spec.update(group='rank_swap', name=name, seed=42, independent_raw=str(RAW),
            independent_ids=read(OLD / 'specs/mrpc_subset0_lambda1.5_s42.json')['independent_ids'],
            replicate=0, model_initialization_seed=42,
            expected_classifier_sha256=HEAD_SHA)
        if name != 'reference':
            spec.update(allocation='fixed', fixed_rank_map=rank_map, swap_id=name)
        job.update(id=ident, cwd=str(frozen), output=str(ROOT / 'runs' / ident),
            log=str(ROOT / 'logs' / f'{ident}.log'))
        job['command'][2] = str(runner)
        for flag,value in (('--output-dir',job['output']),
            ('--extension-spec',ROOT/'specs'/f'{ident}.json'),
            ('--allocation-reference',OLD/'references/mrpc/seed42/allocation.json'),
            ('--training-rng-reference',OLD/'references/mrpc/seed42/training_rng.pt')):
            set_arg(job['command'],flag,value)
        write(ROOT / 'specs' / f'{ident}.json',spec)
        jobs.append(job)
    assert len(jobs) == 13
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in frozen.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    write(ROOT / 'manifest.json', {'created': now(), 'gpu': GPU,
        'scope': 'EXP-02 seed42 reference plus 12 pre-registered rank swaps',
        'wait_for_exp01_complete': True, 'jobs': jobs,
        'pre_registration_sha256': sha(ROOT / 'pre_registered_swaps.json'),
        'rank_allocation_sha256': sha(ROOT / 'swap_rank_allocation.json'),
        'source_hashes': hashes,
        'protocol': 'A0 cached gradients, same seed42 classifier and training RNG, same budget, '
                    'R-Drop and final checkpoint; fixed ranks change only donor and receiver.'})
    write(ROOT / 'status.json', {'updated': now(), 'state': 'prepared', 'total': 13,
        'complete': 0, 'active': None})
    (ROOT / 'PROTOCOL.md').write_text(
        '# Rank-Utility Predictivity protocol\n\n'
        'The 12 swaps are frozen in pre_registered_swaps.json before any swap training. '
        'Historical near-uniform used the nested-prefix calibration set; this run uses A0 '
        'for the reference and every swap, so the reference must be trained again. '
        'Each swap changes exactly two same-shape module ranks by +1 and -1, preserving '
        '755,712 adapter parameters. The frozen SpecCon training runner and Gradient-SVD '
        'initializer are reused. The only support extension installs the pre-registered '
        'rank map and validates the cached-gradient classifier hash. All tasks use '
        'RoBERTa-base, seed42, MRPC, 57 epochs, learning rate 3.5e-4, R-Drop, '
        'same A0 sample IDs and blocks, same restored training RNG, final checkpoint. '
        'The GPU1 controller waits for EXP-01 completion before starting.\n')
    print(now(), 'PREPARED', len(jobs), 'seed42 jobs', flush=True)


def free_memory():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    return {int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}[GPU]


def running_pid(job):
    target=job['output'].encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if b'--output-dir' in args and args[args.index(b'--output-dir')+1]==target:
                return int(proc.name)
        except (OSError,ValueError,IndexError):
            pass
    return None


def validate(job):
    sys.path.insert(0,str(PROJECT/'reproduction'))
    from launch_extended_ablation_20261020 import validate as old_validate
    row=old_validate(job)
    folder=Path(job['output'])
    assert read(folder/'initial_state.json')['classifier_sha256']==HEAD_SHA
    ids=read(folder/'calibration_rows.json')['ids']
    assert ids==job['spec']['independent_ids']
    ranks=read(folder/'allocation.json')['rank_pattern']
    expected=read(ROOT/'swap_rank_allocation.json')
    expected_ranks=expected['reference'] if job['spec']['name']=='reference' else expected['swaps'][job['spec']['name']]
    assert ranks==expected_ranks
    return row


def launch(job):
    env=os.environ.copy()
    for name in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
        env.pop(name,None)
    env.update(CUDA_VISIBLE_DEVICES=str(GPU),PYTHONPATH=job['cwd'],OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='42',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_DISABLED='true')
    Path(job['log']).parent.mkdir(parents=True,exist_ok=True)
    with Path(job['log']).open('a') as log:
        proc=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    return proc.pid


def run():
    with (ROOT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=read(ROOT/'manifest.json')
        assert sha(ROOT/'pre_registered_swaps.json')==manifest['pre_registration_sha256']
        assert sha(ROOT/'swap_rank_allocation.json')==manifest['rank_allocation_sha256']
        for relative,digest in manifest['source_hashes'].items():
            assert sha(ROOT/relative)==digest,relative
        jobs=manifest['jobs']
        completed=read(ROOT/'completed.json') if (ROOT/'completed.json').exists() else {}
        while True:
            if (ROOT/'STOP').exists():
                write(ROOT/'status.json',{'updated':now(),'state':'stopped','total':len(jobs),
                    'complete':len(completed),'active':None})
                return
            prior=read(EXP01/'status.json')
            if prior['state']=='complete' and prior['complete_new']==20:
                break
            if prior['state'] in ('failed','stopped'):
                raise RuntimeError('EXP-01 did not complete; inspect its controller before EXP-02')
            write(ROOT/'status.json',{'updated':now(),'state':'waiting_for_exp01',
                'total':len(jobs),'complete':len(completed),'active':None,
                'exp01_complete_new':prior['complete_new']})
            time.sleep(60)
        while len(completed)<len(jobs):
            if (ROOT/'STOP').exists():
                write(ROOT/'status.json',{'updated':now(),'state':'stopped','total':len(jobs),
                    'complete':len(completed),'active':None})
                return
            job=next(j for j in jobs if j['id'] not in completed)
            folder=Path(job['output'])
            if (folder/'metrics.json').exists():
                completed[job['id']]=validate(job)
                write(ROOT/'completed.json',completed)
                continue
            pid=running_pid(job)
            if pid is None:
                if folder.exists() and any(folder.iterdir()):
                    archive=ROOT/'attempts'/f'{job["id"]}_{int(time.time())}'
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    folder.rename(archive)
                free=free_memory();required=job['min_free_mib']+1024
                if free<required:
                    write(ROOT/'status.json',{'updated':now(),'state':'waiting_for_memory',
                        'total':len(jobs),'complete':len(completed),'active':None,
                        'free_mib':free,'required_mib':required,'next':job['id']})
                    time.sleep(30)
                    continue
                pid=launch(job)
                print(now(),'START',job['id'],'gpu',GPU,'pid',pid,flush=True)
            write(ROOT/'status.json',{'updated':now(),'state':'running','total':len(jobs),
                'complete':len(completed),'active':{'id':job['id'],'pid':pid,'gpu':GPU}})
            while running_pid(job)==pid:
                time.sleep(30)
            if not (folder/'metrics.json').exists():
                raise RuntimeError(f'Training ended before metrics.json: {job["log"]}')
            completed[job['id']]=validate(job)
            write(ROOT/'completed.json',completed)
            print(now(),'DONE',job['id'],completed[job['id']]['score'],flush=True)
            subprocess.check_call([sys.executable,str(PROJECT/'reproduction/analyze_rank_utility_predictivity_v1.py')])
        write(ROOT/'status.json',{'updated':now(),'state':'complete','total':len(jobs),
            'complete':len(completed),'active':None})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run','status'))
    action=parser.parse_args().action
    if action=='prepare': prepare()
    elif action=='run': run()
    else: print(json.dumps(read(ROOT/'status.json'),ensure_ascii=False,indent=2))
