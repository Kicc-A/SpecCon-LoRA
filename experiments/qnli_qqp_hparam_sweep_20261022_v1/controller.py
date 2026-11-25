#!/usr/bin/env python3
"""Audited QNLI/QQP screening, followed by paired three-seed confirmation."""
import argparse
import collections
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

PROJECT = Path('/workspace/SpecCon-LoRA')
ROOT = PROJECT / 'experiments/qnli_qqp_hparam_sweep_20261022_v1'
OLD = PROJECT / 'experiments/extended_ablation_20261021_v2'
PYTHON = '/root/anaconda3/bin/python'
TASKS = ['qnli', 'qqp']
SEEDS = [42, 777, 2026]
LAMBDAS = [0., .1, .25, .5, .7, 1., 1.5, 2., 2.5, 3.]
BASE = {'qnli': dict(lr=.00048, epochs=49, lam=.7, wd=.283, dropout=.05),
        'qqp': dict(lr=.00044, epochs=10, lam=1.5, wd=.173, dropout=0.)}
EPOCHS = {'qnli': [20, 30, 49], 'qqp': [10, 15, 20]}
TRAIN_ROWS = {'qnli': 104743, 'qqp': 363846}
RESERVATION = {'qnli': 10240, 'qqp': 8192}


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(task, lr, epochs, lam):
    ident = f'{task}_lr{lr:.6f}_e{epochs}_lambda{lam:g}'
    return dict(id=ident, task=task, lr=lr, epochs=epochs, lam=lam)


def job_for(cfg, seed, stage='screen', smoke=False):
    ident = cfg['id'] + f'_s{seed}' + ('_smoke' if smoke else '')
    output = ROOT / ('preflight' if smoke else 'runs') / ident
    snapshot = ROOT / 'source_snapshot'
    rng = ROOT / 'references' / 'rng' / cfg['task'] / f'seed{seed}' / 'training_rng.pt'
    base = BASE[cfg['task']]
    command = [PYTHON, '-u', str(snapshot / 'reproduction/run_roberta_other_glue_rdrop.py'),
        '--task', cfg['task'], '--seed', str(seed), '--output-dir', str(output),
        '--learning-rate', str(cfg['lr']), '--epochs', str(cfg['epochs']),
        '--scheduler-epochs', str(cfg['epochs']), '--weight-decay', str(base['wd']),
        '--lora-dropout', str(base['dropout']), '--lr-scheduler', 'linear',
        '--calibration-steps', '16', '--rdrop-alpha', '1.0',
        '--subspace-consistency-lambda', str(cfg['lam']), '--training-rng-reference', str(rng)]
    if smoke:
        command += ['--smoke']
    return dict(id=ident, config=cfg, task=cfg['task'], seed=seed, stage=stage,
                smoke=smoke, output=str(output), command=command, cwd=str(snapshot),
                log=str(ROOT / 'logs' / (ident + '.log')), rng=str(rng))


def result_row(cfg, seed, path, reused=False):
    m = read(Path(path) / 'metrics.json')
    score = m['final_score_percent']
    assert math.isfinite(score)
    return dict(id=cfg['id'] + f'_s{seed}', config=cfg, task=cfg['task'], seed=seed,
                score=score, f1=100 * m['final']['splits']['validation']['f1'] if cfg['task']=='qqp' else None,
                reused=reused, output=str(path), final_epoch=m['epochs'])


def prepare():
    rows = [r for r in read(PROJECT / 'reproduction/current_glue_manifest.json')['runs'] if r['task'] in TASKS]
    original = next(r for r in rows if r['task']=='qnli' and r['seed']==42)
    canonical = Path(original['cwd'])
    assert sha(original['runner']) == original['runner_sha256']
    packages = [{str(f.relative_to(Path(r['cwd']))): sha(f) for f in (Path(r['cwd'])/'speccon_lora').rglob('*.py')} for r in rows]
    assert all(p==packages[0] for p in packages), 'Task packages differ'
    ROOT.mkdir(exist_ok=False)
    for folder in ['logs', 'attempts', 'analysis', 'source_snapshot/reproduction']:
        (ROOT / folder).mkdir(parents=True)
    shutil.copytree(canonical/'speccon_lora', ROOT/'source_snapshot/speccon_lora', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(original['runner'], ROOT/'source_snapshot/reproduction/run_roberta_other_glue_rdrop.py')
    shutil.copy2(__file__, ROOT/'controller.py')
    for row in rows:
        source = Path(row['source'])
        target = ROOT/'references/rng'/row['task']/f"seed{row['seed']}"
        target.mkdir(parents=True)
        for name in ['training_rng.pt', 'training_rng.json']:
            shutil.copy2(source/name, target/name)
    reference_sources = [Path(r['source']) for r in rows]
    for seed in SEEDS:
        reference_sources += [PROJECT/f'experiments/qnli_lambda07_3seeds_gpu0_20261016_v1/runs/qnli/seed{seed}',
            PROJECT/f'experiments/rte_qnli_lambda15_3seeds_gpu012_20261015_v1/runs/qnli/seed{seed}']
    reference_sources += [PROJECT/'experiments/glue_lambda0_seed42_gpu2_20261015_v1/runs/qqp/seed42']
    references = []
    for source in reference_sources:
        m = read(source/'metrics.json'); hp = m['hyperparameters']; task=m['task']; seed=m['seed']
        assert m['state']=='complete' and m['selected']==m['final'] and m['epochs']==BASE[task]['epochs']
        assert m['precision']=='FP32' and m['global_batch_size']==32 and m['adapter_parameters']==755712
        expected = next(r['hyperparameters'] for r in rows if r['task']==task and r['seed']==seed)
        assert {k:v for k,v in hp.items() if k not in ['lambda','training_rng_reference']} == {k:v for k,v in expected.items() if k not in ['lambda','training_rng_reference']}
        rng = read(source/'training_rng.json'); baseline=read(ROOT/'references/rng'/task/f'seed{seed}'/'training_rng.json')
        assert all(rng[k]==baseline[k] for k in ['cpu_sha256','cuda_sha256'])
        cfg=config(task,hp['lr'],hp['epochs'],hp['lambda'])
        target=ROOT/'references/results'/(cfg['id']+f'_s{seed}');target.mkdir(parents=True)
        for name in ['metrics.json','run_config.json','training_rng.json','allocation.json']:
            shutil.copy2(source/name,target/name)
        row=result_row(cfg,seed,target,True);row['original_source']=str(source);references.append(row)
    candidates=[]
    for task in TASKS:
        base=BASE[task]
        for lam in LAMBDAS:
            c=config(task,base['lr'],base['epochs'],lam);c['groups']=['lambda'];candidates.append(c)
        for lr in [base['lr']/2,base['lr']]:
            for epochs in EPOCHS[task]:
                c=config(task,lr,epochs,base['lam']);existing=next((x for x in candidates if x['id']==c['id']),None)
                if existing:
                    existing['groups'].append('lr_epochs')
                else:
                    c['groups']=['lr_epochs'];candidates.append(c)
    reused={r['id'] for r in references}
    jobs=[job_for(c,42) for c in candidates if c['id']+'_s42' not in reused]
    priority={v:i for i,v in enumerate([.25,.5,1.,.1,2.,2.5,3.,.7,0.,1.5])}
    jobs.sort(key=lambda j:(0 if 'lambda' in j['config']['groups'] else 1,
                            priority[j['config']['lam']] if 'lambda' in j['config']['groups'] else j['config']['epochs'],TASKS.index(j['task'])))
    preflights=[job_for(config('qnli',.00048,49,.1),42,'preflight',True),job_for(config('qqp',.00044,10,3.),42,'preflight',True)]
    assert len(candidates)==30 and len(jobs)==25 and len(references)==13
    protocol='''# QNLI / QQP hyperparameter sweep — 2026-10-22

Two distinct screening groups, all using seed 42:

* Lambda only: 0, 0.1, 0.25, 0.5, 0.7, 1, 1.5, 2, 2.5, 3. Keep each task's original LR and epoch budget.
* LR × epochs: QNLI LR {0.00024, 0.00048} × epochs {20,30,49}, lambda=0.7; QQP LR {0.00022,0.00044} × epochs {10,15,20}, lambda=1.5.

The groups overlap at the anchor, giving 15 unique configurations per task, 30 total. Five seed-42 configurations have completed references; 25 new screening runs are queued. These are not the full Cartesian product of all three hyperparameters.

Once all 15 screening configurations for a task succeed, select its two highest seed-42 final-epoch validation accuracies; ties break by fewer epochs, lower LR, lower lambda, then configuration ID. Append seeds 777 and 2026 for each, reusing exact existing matches. At most 8 additional runs, 33 new full runs total. A task with an unresolved screening failure does not silently shortlist an incomplete search. Lambda zero remains a diagnostic candidate; report the strongest nonzero candidate separately. Existing paper/main-table settings are not overwritten.

Every training run starts fresh, with the same per-task/seed saved training RNG, local dataset and RoBERTa-base. FP32, batch32, 755712 trainable adapter parameters, RDrop alpha1, H4, N512, gamma1.2, SVD A/zero B, frozen classifier, linear LR schedule, task-specific unchanged WD/dropout/clip/warmup. Each epoch-budget candidate uses a complete schedule of that length; shorter runs are not truncated old runs. The canonical frozen runner already supports lambda and differs from the older QQP runner only by exposing that existing coefficient and recording it.

Use fixed final epoch Accuracy for selection and comparison; report QQP F1 too. Best checkpoint scores are diagnostic only. Screening rankings and selected three-seed validation results are HPO results, not an untouched test estimate. Report all candidates, n, mean, sample SD and paired deltas. No early stopping. Two smoke runs with full 16-batch calibration must pass before formal launches; smokes never count as full runs.

Scheduling: GPUs 1 and 2 after old experiments on those devices finish. Existing old optimizer processes are never stopped. One new training process per GPU; reserve 10 GiB for QNLI and 8 GiB for QQP, plus 1 GiB headroom. Mean GPU utilization over 3 samples must be <=75% before admission. Borrow GPUs 0/3 only after the old queue reaches a terminal state. On successful completion, release the owned GPU policy. Recover detached children by exact output directory on controller restart. One retry for OOM with increased memory reservation, archive failed output/logs. Other failures remain visible and block that task's automatic shortlist.
'''
    (ROOT/'PROTOCOL.md').write_text(protocol)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ['.py','.json','.pt'] or p.name=='PROTOCOL.md')}
    manifest=dict(created=now(),root=str(ROOT),candidates=candidates,jobs=jobs,preflights=preflights,
                  references=references,source_hashes=hashes,confirmations={},lambda_values=LAMBDAS,
                  max_new_full_runs=33,screening_new_runs=25,gpus=[1,2],seeds=SEEDS)
    write(ROOT/'manifest.json',manifest)
    publish(manifest,{}, {},{}, [],'prepared')
    print(json.dumps(dict(root=str(ROOT),candidates=30,new_screening_runs=25,max_new_full_runs=33)))


def validate(job):
    path=Path(job['output']);m=read(path/'metrics.json');c=job['config'];task=job['task'];hp=m['hyperparameters']
    epochs=1 if job['smoke'] else c['epochs']
    assert m['state']=='complete' and m['task']==task and m['seed']==job['seed']
    assert m['epochs']==epochs and m['planned_epochs']==epochs and not m['stopped_early']
    assert m['selected']==m['final'] and m['final']['epoch']==epochs
    assert m['precision']=='FP32' and m['global_batch_size']==32 and m['adapter_parameters']==755712 and m['gpu_count']==1
    assert m['optimizer_steps']==(16 if job['smoke'] else math.ceil(TRAIN_ROWS[task]/32)*epochs)
    expected=dict(lr=c['lr'],epochs=c['epochs'],wd=BASE[task]['wd'],dropout=BASE[task]['dropout'],
                  scheduler_epochs=epochs,lr_scheduler='linear',calibration_steps=16,calibration_blocks=4,
                  rdrop_alpha=1.,gamma=1.2,scaling=2.,lr_a=c['lr'],lr_b=c['lr'],training_rng_reference=job['rng'])
    expected['lambda']=c['lam']
    for k,v in expected.items():
        assert hp[k]==v,(job['id'],k,hp[k],v)
    rc=read(path/'run_config.json')
    assert rc['source_sha256']==sha(ROOT/'source_snapshot/reproduction/run_roberta_other_glue_rdrop.py')
    assert rc['smoke']==job['smoke'] and rc['subspace_consistency_lambda']==c['lam']
    rng=read(path/'training_rng.json');ref=read(Path(job['rng']).with_suffix('.json'))
    assert all(rng[k]==ref[k] for k in ['cpu_sha256','cuda_sha256'])
    alloc=read(path/'allocation.json');assert len(alloc['target_names'])==72 and alloc['rank_report']['budget_used']==755712
    assert len((path/'history.jsonl').read_text().splitlines())==epochs
    assert abs(m['final_score_percent']-100*m['final']['splits']['validation']['accuracy'])<1e-9
    return result_row(c,job['seed'],path)


def discover(jobs):
    by_output={j['output'].encode():j for j in jobs};found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if b'--output-dir' not in args:continue
            job=by_output.get(args[args.index(b'--output-dir')+1])
            if job is None:continue
            gpu=next(int(v.split(b'=',1)[1]) for v in (p/'environ').read_bytes().split(b'\0') if v.startswith(b'CUDA_VISIBLE_DEVICES='))
            assert job['id'] not in found,'Duplicate process'
            found[job['id']]=dict(pid=int(p.name),gpu=gpu,job=job,adopted=True)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return found


def alive(entry):
    if entry.get('process') is not None:return entry['process'].poll() is None
    try:return entry['job']['output'].encode() in Path(f"/proc/{entry['pid']}/cmdline").read_bytes().split(b'\0')
    except FileNotFoundError:return False


def gpu_status():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(v[0]):dict(free=int(v[1]),utilization=int(v[2])) for line in text.splitlines() if (v:=line.split(','))}


def publish(manifest,complete,failed,active,pending,state,smokes=None,gpus=None,waiting=None):
    records=manifest['references']+list(complete.values())
    groups=[]
    for c in manifest['candidates']:
        rows=sorted([r for r in records if r['config']['id']==c['id']],key=lambda r:r['seed'])
        scores=[r['score'] for r in rows]
        baseline=config(c['task'],BASE[c['task']]['lr'],BASE[c['task']]['epochs'],BASE[c['task']]['lam'])['id']
        base={r['seed']:r['score'] for r in records if r['config']['id']==baseline}
        deltas=[r['score']-base[r['seed']] for r in rows if r['seed'] in base]
        groups.append(dict(**c,n=len(rows),seeds=[r['seed'] for r in rows],seed42=next((r['score'] for r in rows if r['seed']==42),None),
            mean=statistics.mean(scores) if scores else None,sd=statistics.stdev(scores) if len(scores)>1 else None,
            paired_delta=statistics.mean(deltas) if deltas else None,paired_n=len(deltas),
            f1_mean=statistics.mean(r['f1'] for r in rows) if rows and c['task']=='qqp' else None))
    write(ROOT/'results.json',dict(updated=now(),runs=records,groups=groups,confirmations=manifest['confirmations'],failed=failed))
    statuses=[]
    for e in active.values():
        j=e['job'];p=Path(j['output'])/'status.json'
        statuses.append(dict(id=j['id'],pid=e['pid'],gpu=e['gpu'],stage=j['stage'],output=j['output'],training=read(p) if p.exists() else {}))
    write(ROOT/'status.json',dict(state=state,updated=now(),controller_pid=os.getpid(),active=statuses,
          pending=[j['id'] for j in pending],completed=list(complete),failed=failed,smokes=smokes or {},
          new_full_total=len(manifest['jobs']),new_screening_total=25,max_new_full_total=33,
          reused_reference_runs=len(manifest['references']),confirmations=manifest['confirmations'],gpu_status=gpus,waiting=waiting))
    tmp=ROOT/'analysis/candidates.csv.tmp'
    with tmp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    tmp.replace(ROOT/'analysis/candidates.csv')
    lines=['# QNLI / QQP sweep','',f'Updated: {now()}',f'New full runs: {len(complete)}/{len(manifest["jobs"])} complete; {len(failed)} failed. Historical references: 13. Smokes excluded.',
           '', 'All scores are fixed-final-epoch validation Accuracy (%). n=1 is screening only. Paired deltas use matching seeds against QNLI lambda0.7 / QQP lambda1.5 original recipes. QQP F1 is secondary. Hyperparameters are selected on validation data; this is not an untouched test estimate.','']
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    for task in TASKS:
        lines += [f'## {task.upper()}','', '| Group | λ | LR | Epochs | seed42 Acc | n | Mean ± SD | Paired Δ (n) | F1 mean |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g in groups:
            if g['task']!=task:continue
            lines.append(f"| {', '.join(g['groups'])} | {g['lam']:g} | {g['lr']:g} | {g['epochs']} | {fmt(g['seed42'])} | {g['n']} | {fmt(g['mean'])} ± {fmt(g['sd'])} | {fmt(g['paired_delta'])} ({g['paired_n']}) | {fmt(g['f1_mean'])} |")
        ranked=sorted([g for g in groups if g['task']==task and g['seed42'] is not None],key=lambda g:-g['seed42'])
        nonzero=[g for g in ranked if g['lam']>0]
        if ranked:lines += ['',f"Provisional seed42 leader: {ranked[0]['id']} ({fmt(ranked[0]['seed42'])})."]
        if nonzero:lines += [f"Provisional nonzero-lambda leader: {nonzero[0]['id']} ({fmt(nonzero[0]['seed42'])})."]
        lines += ['','Confirmation selection: '+json.dumps(manifest['confirmations'].get(task,'pending all screening results'),ensure_ascii=False),'']
    lines+=['## Failures','',json.dumps(failed,ensure_ascii=False,indent=2)]
    report=ROOT/'analysis/REPORT.md';temp=report.with_suffix('.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(report)


def shortlist(manifest,complete):
    all_results={r['id']:r for r in manifest['references']+list(complete.values())}
    changed=False
    for task in TASKS:
        if task in manifest['confirmations']:continue
        candidates=[c for c in manifest['candidates'] if c['task']==task]
        if not all(c['id']+'_s42' in all_results for c in candidates):continue
        ranked=sorted(candidates,key=lambda c:(-all_results[c['id']+'_s42']['score'],c['epochs'],c['lr'],c['lam'],c['id']))
        winners=ranked[:2]
        manifest['confirmations'][task]=dict(selected_at=now(),configs=[c['id'] for c in winners],
                                            seed42_scores=[all_results[c['id']+'_s42']['score'] for c in winners])
        for c in winners:
            for seed in [777,2026]:
                j=job_for(c,seed,'confirm')
                if j['id'] not in all_results:manifest['jobs'].append(j)
        changed=True
    if changed:write(ROOT/'manifest.json',manifest)


def archive(job):
    output=Path(job['output']);dest=ROOT/'attempts'/(job['id']+'_'+str(time.time_ns()))
    if output.exists():
        output.rename(dest)
        if Path(job['log']).exists():shutil.copy2(job['log'],dest/'process.log')
    return str(dest)


def run():
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(ROOT/'manifest.json')
    for rel,digest in m['source_hashes'].items():assert sha(ROOT/rel)==digest,rel
    all_jobs=m['preflights']+m['jobs'];active=discover(all_jobs);complete={};smokes={}
    failed=read(ROOT/'failed.json') if (ROOT/'failed.json').exists() else {}
    retries=read(ROOT/'retries.json') if (ROOT/'retries.json').exists() else {}
    for job in all_jobs:
        if job['id'] in active:continue
        if (Path(job['output'])/'metrics.json').exists():
            result=validate(job)
            (smokes if job['smoke'] else complete)[job['id']]=result
        elif Path(job['output']).exists() and job['id'] not in failed:
            archive(job)
    samples={gpu:collections.deque(maxlen=3) for gpu in range(4)}
    old_jobs=read(OLD/'manifest.json')['jobs']
    while True:
        for ident,entry in list(active.items()):
            if alive(entry):continue
            job=entry['job'];del active[ident]
            try:
                if entry.get('process') is not None:assert entry['process'].returncode==0,entry['process'].returncode
                result=validate(job);(smokes if job['smoke'] else complete)[ident]=result
                print(now(),'DONE',ident,result['score'],flush=True)
            except Exception as exc:
                log=Path(job['log']).read_text(errors='replace')[-20000:]
                failure=dict(error=repr(exc),time=now(),log=job['log'])
                if ('CUDA out of memory' in log or 'torch.OutOfMemoryError' in log) and retries.get(ident,{}).get('count',0)<1:
                    retries[ident]=dict(count=1,archive=archive(job),failure=failure);write(ROOT/'retries.json',retries)
                else:failed[ident]=failure;write(ROOT/'failed.json',failed)
                print(now(),'FAIL',ident,failure,flush=True)
        shortlist(m,complete)
        all_jobs=m['preflights']+m['jobs'];done=set(complete)|set(smokes)|set(failed)
        pending=[j for j in all_jobs if j['id'] not in done and j['id'] not in active]
        smoke_ok=len(smokes)==len(m['preflights'])
        smoke_failed=any(j['id'] in failed for j in m['preflights'])
        if not pending and not active:
            state='complete' if not failed else 'complete_with_failures'
            publish(m,complete,failed,active,pending,state,smokes)
            policy=OLD/'gpu_policy.json'
            if not failed and policy.exists() and read(policy).get('owner')==ROOT.name:
                write(policy,dict(allowed_gpus=[0,1,2,3],owner=None,released=now(),previous_owner=ROOT.name))
            return
        if smoke_failed and not active:
            publish(m,complete,failed,active,pending,'preflight_failed',smokes);return
        stop=(ROOT/'STOP').exists()
        if stop and not active:
            publish(m,complete,failed,active,pending,'stopped',smokes);return
        try:gpus=gpu_status()
        except (subprocess.SubprocessError,ValueError) as exc:
            print(now(),'GPU_QUERY_RETRY',repr(exc),flush=True);time.sleep(10);continue
        for gpu in samples:samples[gpu].append(gpus[gpu]['utilization'])
        old_active=discover(old_jobs);old_busy={e['gpu'] for e in old_active.values()}
        old_state=read(OLD/'status.json')['state']
        allowed=[1,2]
        if old_state in ['complete','complete_with_failures','stopped'] and not old_active:allowed=[0,1,2,3]
        waiting={}
        for gpu in sorted(allowed,key=lambda g:(gpus[g]['utilization'],-gpus[g]['free'])):
            if stop:waiting[gpu]='STOP file';continue
            if gpu in old_busy:waiting[gpu]='waiting for existing ablation to finish';continue
            if any(e['gpu']==gpu for e in active.values()):continue
            if len(samples[gpu])<3 or statistics.mean(samples[gpu])>75:
                waiting[gpu]='waiting for mean utilization <=75%';continue
            job=next((j for j in pending if (smoke_ok or j['smoke']) and not smoke_failed and
                      gpus[gpu]['free']>=RESERVATION[j['task']]+1024+4096*retries.get(j['id'],{}).get('count',0)),None)
            if job is None:waiting[gpu]='waiting for sufficient memory / preflight';continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED=str(job['seed']),
                     CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
                     HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            assert not Path(job['output']).exists(),job['output']
            with Path(job['log']).open('a') as stream:
                stream.write('\n'+now()+f' LAUNCH physical_gpu={gpu}\n');stream.flush()
                proc=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdout=stream,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            active[job['id']]=dict(pid=proc.pid,process=proc,gpu=gpu,job=job)
            pending.remove(job);print(now(),'START',job['id'],'gpu',gpu,'pid',proc.pid,flush=True)
        publish(m,complete,failed,active,pending,'draining' if stop else 'running',smokes,gpus,waiting)
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','status']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':
        try:run()
        except Exception as exc:
            write(ROOT/'controller_error.json',dict(error=repr(exc),time=now(),pid=os.getpid()));raise
    else:print(json.dumps(read(ROOT/'status.json'),indent=2,ensure_ascii=False))
