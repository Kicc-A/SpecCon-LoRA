#!/usr/bin/env python3
"""Supplement seed2026 using retained best R-Drop configurations on GPUs0/1."""
import argparse, ast, hashlib, json, math, os, shutil, statistics, subprocess, sys, time
from datetime import datetime
from pathlib import Path
P=Path('/workspace/SpecCon-LoRA'); E=P/'experiments'
SMALL=E/'small_glue_rdrop1_rte_mrpc_3seeds_20261003_v1'
OTHER=E/'other_glue_rdrop1_5tasks_3seeds_20261005_v1'
SST=E/'sst2_rdrop_alpha_lr_2configs_3seeds_gpu02_20261007_v1'
COLA=E/'cola_fixed42_rdrop_2configs_3seeds_20261003_v1/runs/RD10/seed2026'
ORDER=('mnli','qnli','qqp','rte','mrpc','stsb','sst2')
SOURCES={t:(SMALL if t in ('rte','mrpc') else SST if t=='sst2' else OTHER) for t in ORDER}
GROUPS={t:('alpha1_lr20' if t=='sst2' else t) for t in ORDER}
TARGETS=dict(cola=63.59,mnli=87.48,mrpc=89.22,qnli=93.01,qqp=90.85,rte=80.02,sst2=94.61,stsb=90.17)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def now():return datetime.now().astimezone().isoformat()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(task,m):
    s=m['final']['splits']
    return 50*(s['matched']['accuracy']+s['mismatched']['accuracy']) if task=='mnli' else 100*s['validation']['matthews_correlation' if task=='cola' else 'pearson' if task=='stsb' else 'accuracy']
def runner(task):return 'run_roberta_small_glue_rdrop.py' if task in ('rte','mrpc') else 'run_roberta_other_glue_rdrop.py'
def command(root,j,out):
    return [sys.executable,'-u',str(root/'source_snapshot/reproduction'/runner(j['task'])),'--task',j['task'],'--seed','2026','--output-dir',str(out),'--epochs',str(j['epochs']),'--scheduler-epochs',str(j['scheduler_epochs']),'--learning-rate',str(j['lr']),'--weight-decay',str(j['wd']),'--lora-dropout',str(j['dropout']),'--lr-scheduler','linear','--calibration-steps','16','--rdrop-alpha',str(j['rdrop_alpha'])]
def env(gpu):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED='2026',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def prepare(root):
    assert root.parent==E
    references={};jobs=[];checks=[]
    for source in {SMALL,OTHER,SST}:
        for name,h in read(source/'launch_manifest.json')['source_sha256'].items():assert digest(source/'source_snapshot'/name)==h,(source,name)
    library=OTHER/'source_snapshot/speccon_lora'
    for f in library.rglob('*.py'):
        for source in (SMALL,SST):assert digest(source/'source_snapshot/speccon_lora'/f.relative_to(library))==digest(f)
    cola=read(COLA/'metrics.json');assert cola['state']=='complete' and cola['seed']==2026 and cola['hyperparameters']['rdrop_alpha']==1
    for task in ORDER:
        records=[]
        for d in sorted((SOURCES[task]/'runs'/GROUPS[task]).glob('seed*')):
            m=read(d/'metrics.json')
            if m.get('state')=='complete':records.append(dict(source=str(d),score=score(task,m),metrics=m))
        assert records and all(r['metrics']['seed']!=2026 for r in records)
        ref=max(records,key=lambda r:r['score']);m=ref['metrics'];hp=m['hyperparameters']
        assert all(r['metrics']['hyperparameters']==hp for r in records)
        assert hp['rdrop_alpha']==1 and hp['epochs']==hp['scheduler_epochs'] and not hp['early_stopping']
        # Read task defaults via AST: values are dict(...) calls containing constants.
        tree=ast.parse((SOURCES[task]/'source_snapshot/reproduction'/runner(task)).read_text())
        cfg_node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TASK_CONFIG' for t in n.targets))
        entry=next(v for k,v in zip(cfg_node.keys,cfg_node.values) if ast.literal_eval(k)==task)
        defaults={k.arg:ast.literal_eval(k.value) for k in entry.keywords}
        assert all(defaults[k]==hp[k] for k in ('clip','max_len','warmup'))
        assert m['adapter_parameters']==755712 and m['global_batch_size']==32
        peak=max(r['metrics']['peak_memory_rank0'] for r in records)/1024**2
        admission=max(13312 if task=='rte' else 10240 if task=='sst2' else 8192,math.ceil(peak)+2048)
        j=dict(task=task,group=GROUPS[task],seed=2026,epochs=hp['epochs'],scheduler_epochs=hp['scheduler_epochs'],lr=hp['lr'],wd=hp['wd'],dropout=hp['dropout'],rdrop_alpha=hp['rdrop_alpha'],steps=m['optimizer_steps'],admission_free_mib=admission)
        jobs.append(j);references[task]=dict(selected_reference=ref,available_completed=records,selection_basis='Highest completed final SST-2 score (three-seed tuning incomplete)' if task=='sst2' else 'Retained best R-Drop configuration; fixed-final scores, no per-seed epoch picking')
        checks.append(dict(task=task,source=ref['source'],hyperparameters=hp,configuration_only_changes=['seed','output_dir','gpu'],passed=True))
    root.mkdir(exist_ok=False);(root/'logs').mkdir();snap=root/'source_snapshot';(snap/'reproduction').mkdir(parents=True)
    shutil.copytree(library,snap/'speccon_lora',ignore=shutil.ignore_patterns('__pycache__'))
    for source,script in ((SMALL,runner('rte')),(OTHER,runner('mnli'))):shutil.copy2(source/'source_snapshot/reproduction'/script,snap/'reproduction'/script)
    assert digest(SST/'source_snapshot/reproduction'/runner('sst2'))==digest(snap/'reproduction'/runner('sst2'))
    shutil.copy2(Path(__file__).resolve(),snap/'reproduction'/Path(__file__).name)
    write(root/'reference_metrics.json',references)
    write(root/'retained_cola_seed2026.json',dict(source=str(COLA),score=score('cola',cola),metrics=cola))
    write(root/'launch_manifest.json',dict(created=now(),jobs=jobs,seeds=[2026],gpus=[0,1],max_concurrent_roberta_jobs_per_gpu=2,selection='Use retained best R-Drop configuration; SST-2 uses alpha1/lr2e-4 with highest completed single-seed final score. CoLA2026 reused.',source_sha256={str(f.relative_to(snap)):digest(f) for f in snap.rglob('*.py')}))
    write(root/'configuration_validation.json',dict(passed=True,checked=now(),checks=checks,frozen_source_verified=True))
    write(root/'status.json',dict(state='prepared',completed=0,total_runs=7,active=[],pending=jobs,failed=[]))
    lines=['# GLUE R-Drop seed2026 补跑','','GPU0/1，按空闲显存准入，每卡最多2个RoBERTa训练进程（含已有任务）；不停止已有训练。','CoLA沿用现有RD10的seed2026。其余7项仅更改seed为2026，复用冻结代码、原轮数和线性衰减。','SST-2选择alpha1、LR2e-4（当前最高完整单seed结果95.0688；三seed调参尚未完成）。','','|任务|训练/衰减轮数|LR|WD|LoRA dropout|R-Drop alpha|','|---|---:|---:|---:|---:|---:|']
    for j in jobs:lines.append(f"|{j['task']}|{j['epochs']}/{j['scheduler_epochs']}|{j['lr']}|{j['wd']}|{j['dropout']}|{j['rdrop_alpha']}|")
    lines+=['','status.json：实时调度状态；summary.json：seed2026及同配置42/777/2026汇总；logs/：各任务日志。','同配置42/777未齐时不计算三seed均值。旧实验仍独立运行。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')

def processes(gpu):
    found=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            args=(d/'cmdline').read_bytes().split(b'\0')
            if not any(Path(a.decode(errors='replace')).name.startswith('run_roberta_') and a.endswith(b'.py') for a in args):continue
            visible=next((s.split(b'=',1)[1].decode() for s in (d/'environ').read_bytes().split(b'\0') if s.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if visible and str(gpu) in visible.split(','):found.append(int(d.name))
        except (OSError,ProcessLookupError):pass
    return found

def validate(root,info):
    out=Path(info['output']);m=read(out/'metrics.json');ref=read(root/'reference_metrics.json')[info['task']]['selected_reference']['metrics']
    assert m['state']=='complete' and m['seed']==2026 and m['task']==info['task']
    assert m['hyperparameters']==ref['hyperparameters']
    assert m['epochs']==m['planned_epochs']==m['final']['epoch']==info['epochs'] and m['optimizer_steps']==info['steps']
    assert not m['stopped_early'] and m['selected']==m['final']
    assert m['adapter_parameters']==755712 and m['global_batch_size']==32 and m['gpu_count']==1 and m['visible_gpus']==str(info['gpu'])
    h=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in h]==list(range(1,info['epochs']+1)) and h[-1]==m['final']
    assert (out/'final_adapter/adapter_model.safetensors').exists()
    cfg=read(out/'final_adapter/adapter_config.json');assert len(cfg['rank_pattern'])==72 and not cfg['use_rslora']
    assert all(cfg['alpha_pattern'][k]==2*r for k,r in cfg['rank_pattern'].items())
    return dict(**info,score=score(info['task'],m),metrics=m)

def summarize(root,completed,active,pending,failed):
    groups=[]
    for task in ('cola',)+ORDER:
        source=COLA.parent if task=='cola' else SOURCES[task]/'runs'/GROUPS[task]
        rows=[]
        for seed in (42,777,2026):
            path=(root/'runs'/task/'seed2026') if task!='cola' and seed==2026 else source/f'seed{seed}'
            m=read(path/'metrics.json')
            if m.get('state')=='complete':rows.append(dict(seed=seed,score=score(task,m),source=str(path)))
        g=dict(task=task,per_seed=rows,complete=len(rows)==3,paper_target=TARGETS[task])
        if len(rows)==3:g.update(mean=statistics.mean(r['score'] for r in rows),sample_std=statistics.stdev(r['score'] for r in rows))
        groups.append(g)
    write(root/'summary.json',dict(updated=now(),seed_set=[42,777,2026],groups=groups,complete=all(g['complete'] for g in groups)))
    write(root/'completed_runs.json',completed)
    state='running' if active else 'failed' if failed else 'waiting_for_gpu_capacity' if pending else 'complete'
    write(root/'status.json',dict(state=state,updated=now(),total_runs=7,completed=len(completed),pending=pending,failed=failed,active=[dict(**i,training_status=read(Path(i['output'])/'status.json')) for _,i in active]))
    if not pending and not active:
        lines=['# Seed2026补跑结果','','|任务|seed2026最终|42/777/2026均值±SD|DPLoRA|','|---|---:|---:|---:|']
        for g in groups:
            s=next((r['score'] for r in g['per_seed'] if r['seed']==2026),None);mean=f"{g['mean']:.4f} ± {g['sample_std']:.4f}" if g['complete'] else '尚未齐'
            lines.append(f"|{g['task']}|{s if s is not None else '失败/缺失'}|{mean}|{g['paper_target']}|")
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def controller(root):
    pending=list(read(root/'launch_manifest.json')['jobs']);active=[];completed=[];failed=[]
    while pending or active:
        running=[]
        for proc,info in active:
            rc=proc.poll()
            if rc is None:running.append((proc,info));continue
            try:
                if rc:raise RuntimeError(f'Exit {rc}; see {info["log"]}')
                completed.append(validate(root,info))
            except Exception as exc:failed.append(dict(**info,error=repr(exc)))
        active=running
        for gpu in (1,0):
            if len(processes(gpu))>=2:continue
            if any(i['gpu']==gpu and read(Path(i['output'])/'status.json').get('state') not in ('running','complete') for _,i in active):continue
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            job=next((j for j in pending if j['admission_free_mib']<=free),None)
            if job is None:continue
            out=root/'runs'/job['task']/'seed2026';log=root/'logs'/f"{job['task']}_seed2026.log";cmd=command(root,job,out)
            with log.open('x') as handle:proc=subprocess.Popen(cmd,cwd=root/'source_snapshot',env=env(gpu),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            info=dict(**job,gpu=gpu,pid=proc.pid,output=str(out),log=str(log),command=cmd,started=now());active.append((proc,info));pending.remove(job);print(json.dumps(info),flush=True)
        summarize(root,completed,active,pending,failed)
        if pending or active:time.sleep(10)
    return 1 if failed else 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--launch-prepared',action='store_true');parser.add_argument('--controller',action='store_true');args=parser.parse_args();root=args.root.resolve()
    if args.controller:
        try:sys.exit(controller(root))
        except Exception as exc:write(root/'controller_error.json',dict(error=repr(exc),updated=now()));raise
    if not args.launch_prepared:prepare(root)
    if not args.prepare_only:
        assert read(root/'status.json')['state']=='prepared' and read(root/'configuration_validation.json')['passed']
        for name,h in read(root/'launch_manifest.json')['source_sha256'].items():assert digest(root/'source_snapshot'/name)==h,name
        with (root/'controller.log').open('x') as handle:proc=subprocess.Popen([sys.executable,'-u',str(root/'source_snapshot/reproduction'/Path(__file__).name),'--root',str(root),'--controller'],cwd=P,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'controller_pid.json',dict(pid=proc.pid));print(json.dumps(dict(root=str(root),pid=proc.pid)))
