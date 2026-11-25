#!/usr/bin/env python3
"""Fixed-final, paired R-Drop transfer to the other five GLUE tasks."""
import argparse,hashlib,json,os,shutil,statistics,subprocess,sys,time
from datetime import datetime
from pathlib import Path
P=Path('/workspace/SpecCon-LoRA')
ORIGIN=P/'experiments/small_glue_rdrop1_rte_mrpc_3seeds_20261003_v1/source_snapshot'
PRECEDING=P/'experiments/small_glue_rdrop1_peak_epochs_49_43_3seeds_20261005_v1'
ORDER=('stsb','sst2','qnli','qqp','mnli')
TARGETS=dict(stsb=90.17,sst2=94.61,qnli=93.01,qqp=90.85,mnli=87.48)
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def now():return datetime.now().astimezone().isoformat()
def score(task,record):
    v=record['splits']
    return 50*(v['matched']['accuracy']+v['mismatched']['accuracy']) if task=='mnli' else 100*v['validation']['pearson' if task=='stsb' else 'accuracy']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(root):
    assert root.parent==P/'experiments'
    selection=read(P/'reports/glue_rank_curves_review_20261001/current_selection.json')
    baseline=[];jobs=[]
    for task in ORDER:
        for r in selection[task]:
            out=Path(r['path']);m=read(out/'metrics.json');hp=m['hyperparameters']
            assert m['state']=='complete' and m['epochs']==hp['epochs']==hp['scheduler_epochs']
            assert hp.get('rdrop_alpha',0)==0 and m['adapter_parameters']==755712
            baseline.append(dict(task=task,seed=r['seed'],source=str(out),score=score(task,m['final']),metrics=m))
            jobs.append(dict(task=task,seed=r['seed'],epochs=m['epochs'],scheduler_epochs=hp['scheduler_epochs'],lr=hp['lr'],
                wd=hp['wd'],dropout=hp['dropout'],rdrop_alpha=1.0,steps=m['final']['step']))
    assert len(jobs)==15 and len({(r['task'],r['seed']) for r in jobs})==15
    root.mkdir(exist_ok=False);(root/'logs').mkdir();snap=root/'source_snapshot';(snap/'reproduction').mkdir(parents=True)
    for name,d in read(ORIGIN.parent/'launch_manifest.json')['source_sha256'].items():assert digest(ORIGIN/name)==d,name
    shutil.copytree(ORIGIN/'speccon_lora',snap/'speccon_lora',ignore=shutil.ignore_patterns('__pycache__'))
    for f in (P/'reproduction/run_roberta_other_glue_rdrop.py',Path(__file__).resolve()):shutil.copy2(f,snap/'reproduction')
    write(root/'baseline_metrics.json',baseline)
    write(root/'launch_manifest.json',dict(created=now(),jobs=jobs,source_origin=str(ORIGIN),seeds=[42,777,2025],
        primary_metrics=dict(stsb='Pearson x100',mnli='mean matched/mismatched accuracy x100',qqp='accuracy x100',qnli='accuracy x100',sst2='accuracy x100'),
        rdrop_alpha=1.0,stsb_consistency='0.5 * mean((prediction1-prediction2)^2); fixed unit variance Gaussian symmetric KL; regression adaptation',
        selection='fixed final epoch; best checkpoint diagnostic only',paper_targets=TARGETS,
        gpu_policy='GPU0 only, up to 2 concurrent jobs; memory-aware admission; GPU1/2/3 unused by this campaign.',
        source_sha256={str(f.relative_to(snap)):digest(f) for f in snap.rglob('*.py')}))
    write(root/'status.json',dict(state='prepared',total_runs=15,pending=jobs,active=[],completed=0,failed=[]))
    (root/'README.md').write_text('# 其余五个 GLUE 任务加入 R-Drop\n\n分类任务使用双向类别KL平均，alpha=1；STS-B为回归适配，监督MSE取两次均值，再加0.5倍两次预测差的均方。\n种子42/777/2025，各任务保留原LR、WD、LoRA dropout、轮数、衰减、warmup、clip、校准和参数预算。\n轮数：STS-B67、SST-2 38、QNLI49、QQP10、MNLI32。不早停，最终轮比较。\n固定同一步数、每步两次前向，计算量高于原基线。\n主指标：STS-B Pearson，MNLI matched/mismatched准确率均值，其他accuracy，均乘100。\n用户指定GPU0；最多2个任务并行，按显存准入。GPU1/2/3不用于本批任务。\n原始对照路径及指标见baseline_metrics.json。STS-B不是对单logit做softmax KL（该项恒为0）。\n')

def command(root,job,out):
    return [sys.executable,'-u',str(root/'source_snapshot/reproduction/run_roberta_other_glue_rdrop.py'),
        '--task',job['task'],'--seed',str(job['seed']),'--output-dir',str(out),'--epochs',str(job['epochs']),
        '--scheduler-epochs',str(job['scheduler_epochs']),'--learning-rate',str(job['lr']),'--weight-decay',str(job['wd']),
        '--lora-dropout',str(job['dropout']),'--lr-scheduler','linear','--calibration-steps','16','--rdrop-alpha','1.0']
def env(gpu,seed):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED=str(seed),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def validate(info,baseline):
    out=Path(info['output']);m=read(out/'metrics.json');hp=m['hyperparameters']
    assert m['state']=='complete' and (m['task'],m['seed'])==(info['task'],info['seed'])
    assert m['epochs']==m['planned_epochs']==m['final']['epoch']==info['epochs']
    assert hp['scheduler_epochs']==info['scheduler_epochs'] and m['optimizer_steps']==info['steps']
    assert not m['stopped_early'] and not hp['early_stopping'] and m['selected']==m['final']
    for k,v in baseline['metrics']['hyperparameters'].items():assert hp[k]==v,(k,hp.get(k),v)
    assert hp['rdrop_alpha']==1 and hp['label_smoothing']==0 and hp['training_forwards_per_step']==2
    expected='unit_variance_gaussian_symmetric_kl_half_mse' if info['task']=='stsb' else 'class_sum_batch_mean_bidirectional_average'
    assert hp['rdrop_kl_reduction']==expected
    assert m['adapter_parameters']==755712 and m['global_batch_size']==32 and m['gpu_count']==1 and m['visible_gpus']==str(info['gpu'])
    assert abs(m['final']['train_loss']-m['final']['train_supervised_loss']-m['final']['train_consistency_loss'])<1e-5
    assert m['final']['train_consistency_loss']>=-1e-6
    h=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in h]==list(range(1,info['epochs']+1)) and h[-1]==m['final']
    assert read(out/'allocation.json')['rank_pattern']==read(Path(baseline['source'])/'allocation.json')['rank_pattern']
    assert (out/'final_adapter/adapter_model.safetensors').exists()
    cfg=read(out/'final_adapter/adapter_config.json');assert len(cfg['rank_pattern'])==72 and not cfg['use_rslora']
    assert all(cfg['alpha_pattern'][k]==2*r for k,r in cfg['rank_pattern'].items())
    value=score(info['task'],m['final']);assert abs(value-m['final_score_percent'])<1e-8
    return dict(**info,score=value,reference_score=baseline['score'],delta=value-baseline['score'],metrics=m)

def summarize(root,completed,active,pending,failed):
    baseline=read(root/'baseline_metrics.json');groups=[]
    for task in ORDER:
        old=[r for r in baseline if r['task']==task];done=[r for r in completed if r['task']==task]
        g=dict(task=task,complete=len(done)==3,baseline_mean=statistics.mean(r['score'] for r in old),baseline_sample_std=statistics.stdev(r['score'] for r in old),
            paper_target=TARGETS[task],per_seed=[dict(seed=r['seed'],score=r['score'],reference_score=r['reference_score'],delta=r['delta']) for r in done])
        if len(done)==3:g.update(final_mean=statistics.mean(r['score'] for r in done),final_sample_std=statistics.stdev(r['score'] for r in done),mean_delta=statistics.mean(r['delta'] for r in done))
        groups.append(g)
    write(root/'summary.json',dict(updated=now(),complete=len(completed)==15,groups=groups))
    write(root/'completed_runs.json',completed)
    state='running' if active else 'failed' if failed else 'waiting_for_gpu' if pending else 'complete'
    write(root/'status.json',dict(state=state,updated=now(),total_runs=15,completed=len(completed),pending=pending,failed=failed,
        active=[dict(**info,training_status=read(Path(info['output'])/'status.json')) for _,info in active]))
    if len(completed)==15:
        lines=['# R-Drop 其他GLUE任务结果','','固定最终轮，三seed均值±样本SD；STS-B是Gaussian回归一致性适配。','',
            '| 任务 | 原基线 | 本次 | 差异 |','|---|---:|---:|---:|']
        for g in groups:lines.append(f"| {g['task']} | {g['baseline_mean']:.4f} ± {g['baseline_sample_std']:.4f} | {g['final_mean']:.4f} ± {g['final_sample_std']:.4f} | {g['mean_delta']:+.4f} |")
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def controller(root):
    pending=list(read(root/'launch_manifest.json')['jobs']);active=[];completed=[];failed=[]
    baseline={(r['task'],r['seed']):r for r in read(root/'baseline_metrics.json')}
    try:
        while pending or active:
            running=[]
            for process,info in active:
                rc=process.poll()
                if rc is None:running.append((process,info));continue
                try:
                    if rc:raise RuntimeError(f'Exit {rc}; see {info["log"]}')
                    completed.append(validate(info,baseline[(info['task'],info['seed'])]))
                except Exception as exc:failed.append(dict(**info,error=repr(exc)))
            active=running
            if not failed:
                for gpu in (0,0):
                    if not pending or sum(info['gpu']==gpu for _,info in active)>=2:continue
                    if any(read(Path(info['output'])/'status.json').get('state') not in ('running','complete') for _,info in active):continue
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    reserve=13312 if pending[0]['task']=='sst2' else 8192
                    if free<reserve+2048:continue
                    job=pending.pop(0);out=root/'runs'/job['task']/f"seed{job['seed']}";log=root/'logs'/f"{job['task']}_seed{job['seed']}.log"
                    cmd=command(root,job,out)
                    with log.open('x') as f:process=subprocess.Popen(cmd,cwd=root/'source_snapshot',env=env(gpu,job['seed']),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                    info=dict(**job,gpu=gpu,pid=process.pid,output=str(out),log=str(log),command=cmd,started=now());active.append((process,info));print(json.dumps(info),flush=True)
            summarize(root,completed,active,pending,failed)
            if failed and not active:return 1
            if pending or active:time.sleep(10)
        return 0
    except Exception as exc:
        write(root/'controller_error.json',dict(error=repr(exc),updated=now(),active=[i for _,i in active],pending=pending));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--launch-prepared',action='store_true');parser.add_argument('--controller',action='store_true');args=parser.parse_args();root=args.root.resolve()
    if args.controller:sys.exit(controller(root))
    if not args.launch_prepared:prepare(root)
    if not args.prepare_only:
        assert read(root/'status.json')['state']=='prepared' and read(root/'preflight_validation.json')['passed']
        for name,h in read(root/'launch_manifest.json')['source_sha256'].items():assert digest(root/'source_snapshot'/name)==h,name
        with (root/'controller.log').open('x') as f:
            proc=subprocess.Popen([sys.executable,'-u',str(root/'source_snapshot/reproduction'/Path(__file__).name),'--root',str(root),'--controller'],cwd=P,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'controller_pid.json',dict(pid=proc.pid));print(json.dumps(dict(root=str(root),pid=proc.pid)))
