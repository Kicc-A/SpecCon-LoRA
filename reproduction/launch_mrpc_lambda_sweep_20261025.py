#!/usr/bin/env python3
"""Frozen MRPC lambda screening with accuracy-based, paired-seed confirmation."""
import argparse
import ast
import collections
from datetime import datetime
import difflib
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
import traceback

PROJECT = Path('/workspace/SpecCon-LoRA')
ROOT = PROJECT / 'experiments/mrpc_lambda_sweep_20261025_v1'
PYTHON = '/root/anaconda3/bin/python'
LAMBDAS = [.25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]
SEEDS = [42, 777, 2026]
TRAIN_ROWS = 3668
GPU = 0
MEMORY_RESERVE_MIB = 9216
HEADROOM_MIB = 1024


def now(): return datetime.now().astimezone().isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def expose_lambda(original):
    replacements = [
        ('    args = parser.parse_args()\n',
         '    parser.add_argument("--subspace-consistency-lambda", type=float, default=1.5)\n'
         '    args = parser.parse_args()\n'
         '    if not math.isfinite(args.subspace_consistency_lambda) or args.subspace_consistency_lambda < 0:\n'
         '        parser.error("--subspace-consistency-lambda must be finite and nonnegative")\n'),
        ('subspace_consistency_lambda=1.5,', 'subspace_consistency_lambda=args.subspace_consistency_lambda,'),
        ('"gamma": 1.2, "lambda": 1.5,', '"gamma": 1.2, "lambda": args.subspace_consistency_lambda,'),
        ('        "smoke": args.smoke, "scheduler_epochs": cfg["scheduler_epochs"],\n',
         '        "smoke": args.smoke, "scheduler_epochs": cfg["scheduler_epochs"],\n'
         '        "subspace_consistency_lambda": args.subspace_consistency_lambda,\n'),
    ]
    updated = original
    for old, new in replacements:
        assert updated.count(old) == 1, old
        updated = updated.replace(old, new)
    before = {n.name: ast.dump(n) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in ast.parse(updated).body if isinstance(n, ast.FunctionDef)}
    assert before.keys() == after.keys()
    assert all(before[k] == after[k] for k in before if k not in {'parse_args', 'main'})
    return updated


def job_for(lam, seed, stage='screen', smoke=False):
    ident = f'mrpc_lambda{lam:g}_s{seed}' + ('_smoke' if smoke else '')
    snapshot = ROOT / 'source_snapshot'
    output = ROOT / ('preflight' if smoke else 'runs') / ident
    rng = ROOT / 'references' / f'seed{seed}' / 'training_rng.pt'
    command = [PYTHON, '-u', str(snapshot / 'reproduction/run_roberta_small_glue_regularization.py'),
        '--task', 'mrpc', '--seed', str(seed), '--output-dir', str(output),
        '--epochs', '57', '--scheduler-epochs', '57', '--learning-rate', '0.00035',
        '--weight-decay', '0.307', '--lora-dropout', '0.05', '--lr-scheduler', 'linear',
        '--calibration-steps', '16', '--rdrop-alpha', '1.0', '--label-smoothing', '0.05',
        '--subspace-consistency-lambda', str(lam), '--training-rng-reference', str(rng)]
    if smoke: command += ['--smoke', '--smoke-epochs', '1']
    return dict(id=ident, task='mrpc', lam=lam, seed=seed, stage=stage, smoke=smoke,
        output=str(output), log=str(ROOT / 'logs' / (ident + '.log')),
        command=command, cwd=str(snapshot), rng=str(rng))


def result_row(job, reused=False, original_source=None):
    metric = read(Path(job['output']) / 'metrics.json')
    validation = metric['final']['splits']['validation']
    # MRPC final_score_percent is Accuracy/F1 mean; never use it for ranking.
    accuracy = 100 * validation['accuracy']; f1 = 100 * validation['f1']
    assert math.isfinite(accuracy) and math.isfinite(f1)
    assert abs(accuracy - metric['final_accuracy_percent']) < 1e-9
    return dict(id=job['id'], task='mrpc', lam=job['lam'], seed=job['seed'], stage=job['stage'],
        accuracy=accuracy, f1=f1, combined=100 * validation['combined'],
        epochs=metric['epochs'], output=job['output'], reused=reused, original_source=original_source)


def prepare():
    source_rows = [r for r in read(PROJECT / 'reproduction/current_glue_manifest.json')['runs'] if r['task']=='mrpc']
    assert {r['seed'] for r in source_rows} == set(SEEDS)
    source_rows.sort(key=lambda r: SEEDS.index(r['seed']))
    for row in source_rows:
        assert sha(row['runner']) == row['runner_sha256']
        metric = read(Path(row['source']) / 'metrics.json')
        assert metric['state']=='complete' and metric['epochs']==57 and metric['selected']==metric['final']
        assert metric['hyperparameters']==row['hyperparameters']
        assert metric['adapter_parameters']==755712 and metric['global_batch_size']==32
        assert row['hyperparameters']['gamma']==1.2 and row['hyperparameters']['lambda']==1.5
    assert len({r['runner_sha256'] for r in source_rows})==1
    base = source_rows[0]
    original = Path(base['runner']).read_text(); modified = expose_lambda(original)
    packages = [{str(p.relative_to(Path(r['cwd']))): sha(p) for p in (Path(r['cwd'])/'speccon_lora').rglob('*.py')} for r in source_rows]
    assert all(p==packages[0] for p in packages)
    ROOT.mkdir(exist_ok=False)
    for name in ['logs','attempts','analysis','source_snapshot/reproduction','references']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(base['cwd'])/'speccon_lora', ROOT/'source_snapshot/speccon_lora', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (ROOT/'source_snapshot/reproduction/run_roberta_small_glue_regularization.py').write_text(modified)
    shutil.copy2(__file__, ROOT/'controller.py')
    shutil.copy2(PROJECT/'reproduction/current_glue_manifest.json', ROOT/'original_manifest.json')
    (ROOT/'references/original_runner.py').write_text(original)
    (ROOT/'references/runner.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile=base['runner'],tofile='source_snapshot/reproduction/run_roberta_small_glue_regularization.py')))
    references=[]
    for row in source_rows:
        ref=ROOT/'references'/f"seed{row['seed']}";ref.mkdir()
        for name in ['metrics.json','run_config.json','allocation.json','training_rng.json','training_rng.pt']:
            shutil.copy2(Path(row['source'])/name,ref/name)
        j=dict(id=f"mrpc_lambda1.5_s{row['seed']}",lam=1.5,seed=row['seed'],stage='reference',output=str(ref))
        references.append(result_row(j,True,row['source']))
    jobs=[job_for(lam,42) for lam in LAMBDAS if lam != 1.5]
    preflights=[job_for(lam,42,'preflight',True) for lam in [.25,2.]]
    protocol='''# MRPC λ 扫描：2026-10-25

扫描 λ=0.25、0.50、0.75、1.00、1.25、1.50、1.75、2.00。先用 seed42 筛选，λ=1.5 的三种子参考已完成且协议一致；新增筛选为7次。
所有8个seed42结果成功后，按固定末轮验证Accuracy降序选择前2个配置；相同分数优先较小λ。
对选中配置补seed777、2026，已有完全匹配结果复用，至多再新增4次。筛选失败不从不完整集合挑选。

固定 RoBERTa-base、FP32、batch32、755712适配器参数、57训练/调度轮数、学习率0.00035、γ=1.2、H4、N512、
SVD A/零B、scaling2、R-Drop α=1、label smoothing0.05、LoRA dropout0.05、WD0.307、clip1.498、
max_len128、warmup0.038、linear scheduler、冻结分类头。每次从基座重新训练，只复用各训练种子的CPU/CUDA随机状态，不复用已训练权重。
唯一训练变量是λ。源码快照仅为原脚本暴露已有λ参数、校验和记录；其他辅助函数AST完全一致。

以 final.splits.validation.accuracy×100 排序，并报告F1与Accuracy/F1均值。
原MRPC脚本的final_score_percent是Accuracy/F1均值，不能当Accuracy。固定末轮；最佳checkpoint仅诊断，不参与筛选。
报告每个候选的n、mean±样本SD、匹配种子的配对差值；验证集HPO不作为未参与选择的测试集估计。

两个端点先做完整16-batch校准与单轮512样本预检，均通过后才投放正式训练；预检不计新增正式训练。
默认GPU0，最多1个本队列任务。仅在连续3次采样平均利用率≤75%、空闲显存≥9GiB+1GiB余量时投放；
允许与已有QQP共享GPU0的剩余容量，不停止其他进程，也不改变已有队列。并发时耗时不作公平速度比较。
OOM保留attempt并至多重试1次，重试显存门槛提高2GiB；其他错误记录，不自动扩大扫描或无限重试。
控制器可重启并按精确output路径接管存活训练。STOP文件只阻止新投放，正在训练的任务正常完成。
GPU0临时不可用时保持后台等待，并在status.json说明原因。
'''
    (ROOT/'PROTOCOL.md').write_text(protocol)
    protected=[ROOT/'controller.py',ROOT/'PROTOCOL.md',ROOT/'original_manifest.json']+list((ROOT/'source_snapshot').rglob('*.py'))+list((ROOT/'references').rglob('*'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    manifest=dict(created=now(),lambda_values=LAMBDAS,seeds=SEEDS,screen_seed=42,primary_metric='fixed-final validation Accuracy x100',
        references=references,jobs=jobs,preflights=preflights,confirmation=None,source_hashes=hashes,
        base_hyperparameters=base['hyperparameters'],screening_new_runs=7,max_new_full_runs=11,
        scheduling=dict(gpu=GPU,max_active=1,reserve_mib=MEMORY_RESERVE_MIB,headroom_mib=HEADROOM_MIB,max_mean_utilization=75,samples=3,poll_seconds=10))
    write(ROOT/'manifest.json',manifest)
    publish(manifest,{}, {},{}, {},preflights+jobs,'prepared',waiting='Not launched yet')
    print(json.dumps(dict(root=str(ROOT),grid=LAMBDAS,new_screening=7,max_new_full=11,references=3),ensure_ascii=False))


def validate(job, manifest):
    path=Path(job['output']);metric=read(path/'metrics.json')
    epochs=1 if job['smoke'] else 57
    assert metric['state']=='complete' and metric['task']=='mrpc' and metric['seed']==job['seed']
    assert metric['epochs']==metric['planned_epochs']==epochs and not metric['stopped_early']
    assert metric['selected']==metric['final'] and metric['final']['epoch']==epochs
    assert metric['precision']=='FP32' and metric['global_batch_size']==32 and metric['gpu_count']==1
    assert metric['adapter_parameters']==755712 and metric['optimizer_steps']==(16 if job['smoke'] else math.ceil(TRAIN_ROWS/32)*57)
    expected=dict(manifest['base_hyperparameters'])
    expected.update({'lambda':job['lam'],'training_rng_reference':job['rng'],'scheduler_epochs':epochs})
    assert metric['hyperparameters']==expected,(job['id'],metric['hyperparameters'],expected)
    assert len((path/'history.jsonl').read_text().splitlines())==epochs
    cfg=read(path/'run_config.json')
    assert cfg['source_sha256']==sha(ROOT/'source_snapshot/reproduction/run_roberta_small_glue_regularization.py')
    assert cfg['smoke']==job['smoke'] and cfg['subspace_consistency_lambda']==job['lam']
    rng=read(path/'training_rng.json');ref=read(Path(job['rng']).with_suffix('.json'))
    assert all(rng[k]==ref[k] for k in ['cpu_sha256','cuda_sha256'])
    alloc=read(path/'allocation.json')
    assert len(alloc['target_names'])==len(alloc['rank_pattern'])==72 and alloc['rank_report']['budget_used']==755712
    return result_row(job)


def publish(m,complete,smokes,failed,active,pending,state,gpu=None,waiting=None):
    records=m['references']+list(complete.values());base={r['seed']:r['accuracy'] for r in m['references']}
    groups=[]
    for lam in LAMBDAS:
        rs=sorted([r for r in records if r['lam']==lam],key=lambda r:r['seed'])
        scores=[r['accuracy'] for r in rs];ds=[r['accuracy']-base[r['seed']] for r in rs]
        groups.append(dict(lam=lam,n=len(rs),seeds=[r['seed'] for r in rs],seed42=next((r['accuracy'] for r in rs if r['seed']==42),None),
            mean=statistics.mean(scores) if scores else None,sd=statistics.stdev(scores) if len(scores)>1 else None,
            f1_mean=statistics.mean(r['f1'] for r in rs) if rs else None,
            paired_delta=statistics.mean(ds) if ds else None,paired_n=len(ds)))
    write(ROOT/'results.json',dict(updated=now(),runs=records,groups=groups,confirmation=m['confirmation'],failed=failed))
    states=[]
    for e in active.values():
        j=e['job'];p=Path(j['output'])/'status.json'
        states.append(dict(id=j['id'],pid=e['pid'],gpu=e['gpu'],stage=j['stage'],output=j['output'],training=read(p) if p.exists() else {}))
    write(ROOT/'status.json',dict(state=state,updated=now(),controller_pid=os.getpid(),active=states,
        completed=list(complete),new_full_total=len(m['jobs']),screening_new_total=7,max_new_full_total=11,
        pending=[j['id'] for j in pending],preflights_passed=list(smokes),preflights_total=2,failed=failed,
        reused_references=3,confirmation=m['confirmation'],gpu=gpu,waiting=waiting))
    import csv
    tmp=ROOT/'analysis/candidates.csv.tmp'
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
    tmp.replace(ROOT/'analysis/candidates.csv')
    fmt=lambda v:'—' if v is None else f'{v:.4f}'
    lines=['# MRPC λ 扫描','',f'更新：{now()}。状态：{state}。',
        f'新增正式训练 {len(complete)}/{len(m["jobs"])}；预检 {len(smokes)}/2；失败 {len(failed)}；另复用 λ=1.5 的3个种子。',
        '固定末轮验证Accuracy×100为主指标；F1另列。n=1为筛选，n=3为配对种子复核。±为样本SD。',
        '所有配置：γ=1.2，LR=0.00035，57轮，755712参数；只改变λ。配对差值为当前配置减同种子的λ=1.5。',
        '', '| λ | seed42 Accuracy | n | Accuracy mean ± SD | F1 mean | 配对Δ (n) |','|---:|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['lam']:g} | {fmt(g['seed42'])} | {g['n']} | {fmt(g['mean'])} ± {fmt(g['sd'])} | {fmt(g['f1_mean'])} | {fmt(g['paired_delta'])} ({g['paired_n']}) |")
    ranked=sorted([g for g in groups if g['seed42'] is not None],key=lambda g:(-g['seed42'],g['lam']))
    if ranked:lines+=['',f"当前seed42最高：λ={ranked[0]['lam']:g}，Accuracy={fmt(ranked[0]['seed42'])}。候选尚未齐全时仅为临时排名。"]
    lines+=['','多种子确认：'+json.dumps(m['confirmation'],ensure_ascii=False), '','资源等待原因：'+str(waiting), '', '实验协议见 ../PROTOCOL.md。训练日志与原始指标分别在 logs/ 与 runs/。']
    p=ROOT/'analysis/REPORT.md';tmp=p.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(p)


def discover(jobs):
    by_output={j['output'].encode():j for j in jobs};found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if b'--output-dir' not in args:continue
            job=by_output.get(args[args.index(b'--output-dir')+1])
            if not job:continue
            gpu=next(int(x.split(b'=',1)[1]) for x in (p/'environ').read_bytes().split(b'\0') if x.startswith(b'CUDA_VISIBLE_DEVICES='))
            assert job['id'] not in found,'Duplicate output process'
            found[job['id']]=dict(job=job,gpu=gpu,pid=int(p.name),adopted=True)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return found


def alive(entry):
    if 'process' in entry:return entry['process'].poll() is None
    try:return entry['job']['output'].encode() in Path(f"/proc/{entry['pid']}/cmdline").read_bytes().split(b'\0')
    except FileNotFoundError:return False


def archive(job):
    p=Path(job['output']);dest=ROOT/'attempts'/(job['id']+'_'+str(time.time_ns()))
    if p.exists():
        p.rename(dest)
        if Path(job['log']).exists():shutil.copy2(job['log'],dest/'process.log')
    return str(dest)


def shortlist(m,complete):
    if m['confirmation'] is not None:return
    records={r['id']:r for r in m['references']+list(complete.values())}
    if not all(f'mrpc_lambda{x:g}_s42' in records for x in LAMBDAS):return
    ranked=sorted(LAMBDAS,key=lambda x:(-records[f'mrpc_lambda{x:g}_s42']['accuracy'],x))[:2]
    m['confirmation']=dict(selected_at=now(),lambdas=ranked,seed42_scores=[records[f'mrpc_lambda{x:g}_s42']['accuracy'] for x in ranked])
    for lam in ranked:
        for seed in [777,2026]:
            job=job_for(lam,seed,'confirm')
            if job['id'] not in records:m['jobs'].append(job)
    write(ROOT/'manifest.json',m)


def run():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(ROOT/'manifest.json')
    for rel,digest in m['source_hashes'].items():assert sha(ROOT/rel)==digest,rel
    active=discover(m['preflights']+m['jobs']);complete={};smokes={}
    failed=read(ROOT/'failed.json') if (ROOT/'failed.json').exists() else {}
    retries=read(ROOT/'retries.json') if (ROOT/'retries.json').exists() else {}
    for job in m['preflights']+m['jobs']:
        if job['id'] in active or job['id'] in failed:continue
        if (Path(job['output'])/'metrics.json').exists():
            row=validate(job,m);(smokes if job['smoke'] else complete)[job['id']]=row
        elif Path(job['output']).exists():archive(job)
    samples=collections.deque(maxlen=3)
    while True:
        for ident,e in list(active.items()):
            if alive(e):continue
            j=e['job'];del active[ident]
            try:
                if 'process' in e:assert e['process'].returncode==0,e['process'].returncode
                row=validate(j,m);(smokes if j['smoke'] else complete)[ident]=row
                print(now(),'DONE',ident,'Accuracy',row['accuracy'],'F1',row['f1'],flush=True)
            except Exception as exc:
                log=Path(j['log']).read_text(errors='replace')[-20000:] if Path(j['log']).exists() else ''
                error=dict(time=now(),error=repr(exc),log=j['log'])
                if ('CUDA out of memory' in log or 'torch.OutOfMemoryError' in log) and retries.get(ident,{}).get('count',0)<1:
                    retries[ident]=dict(count=1,archive=archive(j),error=error);write(ROOT/'retries.json',retries)
                else:failed[ident]=error;write(ROOT/'failed.json',failed)
                print(now(),'FAIL',ident,error,flush=True)
        shortlist(m,complete)
        done=set(complete)|set(smokes)|set(failed)
        pending=[j for j in m['preflights']+m['jobs'] if j['id'] not in done and j['id'] not in active]
        smoke_ok=len(smokes)==2;smoke_failed=any(j['id'] in failed for j in m['preflights'])
        stop=(ROOT/'STOP').exists()
        if (not pending and not active) or (smoke_failed and not active) or (stop and not active):
            state='preflight_failed' if smoke_failed else ('stopped' if stop else ('complete_with_failures' if failed else 'complete'))
            publish(m,complete,smokes,failed,active,pending,state);return
        gpu=None;waiting=None
        try:
            raw=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            free,util=map(int,raw.strip().split(','));samples.append(util)
            gpu=dict(index=GPU,free_mib=free,utilization=util,mean_utilization=statistics.mean(samples),sample_count=len(samples))
        except (OSError,subprocess.SubprocessError,ValueError) as exc:waiting='GPU query failed: '+repr(exc)
        if active:waiting='One MRPC job is already active'
        elif stop:waiting='STOP file; no new admission'
        elif gpu is not None:
            if len(samples)<3 or statistics.mean(samples)>75:waiting='Waiting for 3-sample GPU0 mean utilization <=75%'
            else:
                job=next((j for j in pending if (smoke_ok or j['smoke']) and not smoke_failed),None)
                required=MEMORY_RESERVE_MIB+HEADROOM_MIB+2048*retries.get(job['id'],{}).get('count',0) if job else None
                if job is None:waiting='Waiting for preflight gate'
                elif free<required:waiting=f'Waiting for GPU0 free memory >= {required} MiB (current {free})'
                else:
                    assert not Path(job['output']).exists(),job['output']
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU),PYTHONHASHSEED=str(job['seed']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                    with Path(job['log']).open('a') as stream:
                        stream.write('\n'+now()+f' LAUNCH physical_gpu={GPU}\n');stream.flush()
                        proc=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    active[job['id']]=dict(pid=proc.pid,process=proc,gpu=GPU,job=job)
                    pending.remove(job);samples.clear();waiting=None
                    print(now(),'START',job['id'],'GPU',GPU,'PID',proc.pid,flush=True)
        publish(m,complete,smokes,failed,active,pending,'draining' if stop else ('running' if active else 'waiting_for_resources'),gpu,waiting)
        time.sleep(10)


def launch():
    existing=ROOT/'launch.json'
    if existing.exists():
        prior=read(existing);p=Path('/proc')/str(prior['pid'])/'cmdline'
        if p.exists() and str(ROOT/'controller.py').encode() in p.read_bytes().split(b'\0'):
            raise RuntimeError(f"Controller already running: {prior['pid']}")
    with (ROOT/'controller.log').open('a') as stream:
        proc=subprocess.Popen([PYTHON,'-u',str(ROOT/'controller.py'),'run'],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=proc.pid,started=now(),controller=str(ROOT/'controller.py'))
    write(existing,record);print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','launch','run','status']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='launch':launch()
    elif args.action=='status':print(json.dumps(read(ROOT/'status.json'),ensure_ascii=False,indent=2))
    else:
        try:run()
        except Exception as exc:
            write(ROOT/'controller_error.json',dict(time=now(),error=repr(exc),pid=os.getpid(),traceback=traceback.format_exc()))
            status=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
            status.update(state='controller_error',updated=now(),error=repr(exc));write(ROOT/'status.json',status)
            raise
