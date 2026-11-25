#!/usr/bin/env python3
import argparse,concurrent.futures,copy,hashlib,importlib.util,json,os,shutil,statistics,subprocess,sys,time
from datetime import datetime
from pathlib import Path
P=Path('/workspace/SpecCon-LoRA');ROOT=P/'experiments/small_glue_rdrop_dropout_labelsmoothing_3seeds_gpu23_20261008_v1';OLD=P/'experiments/small_glue_rdrop1_rte_mrpc_3seeds_20261003_v1';SUP=P/'experiments/glue_rdrop_best_seed2026_gpu01_20261008_v1'
CONFIGS=dict(rte=dict(gpu=2,epochs=63,lr=.00021,wd=.198,dropout=0.,steps=4914),mrpc=dict(gpu=3,epochs=57,lr=.00035,wd=.307,dropout=.05,steps=6555))
GROUPS=[dict(task=task,group=task+'_'+ablation,ablation=ablation,label_smoothing=.05 if ablation=='ls005' else 0.,**{**cfg,'dropout':(.05 if task=='rte' else .10) if ablation=='dropout' else cfg['dropout']}) for task,cfg in CONFIGS.items() for ablation in ('dropout','ls005')]
JOBS=[dict(**g,seed=seed,scheduler_epochs=g['epochs'],rdrop_alpha=1.) for seed in (42,777,2026) for g in GROUPS]
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def write(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def now():return datetime.now().astimezone().isoformat()
def env(gpu,seed):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED=str(seed),CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def command(root,j,out,reference=False,smoke_epochs=None):
 script='reference_small_glue_rdrop.py' if reference else 'run_roberta_small_glue_regularization.py'
 c=[sys.executable,'-u',str(root/'source_snapshot/reproduction'/script),'--task',j['task'],'--seed',str(j['seed']),'--output-dir',str(out),'--epochs',str(j['epochs']),'--scheduler-epochs',str(j['scheduler_epochs']),'--learning-rate',str(j['lr']),'--weight-decay',str(j['wd']),'--lora-dropout',str(j['dropout']),'--lr-scheduler','linear','--calibration-steps','4' if smoke_epochs else '16','--rdrop-alpha','1.0']
 if not reference:c+=['--label-smoothing',str(j['label_smoothing'])]
 if smoke_epochs:c+=['--smoke','--smoke-epochs',str(smoke_epochs)]
 return c

def prepare(root):
 refs=[]
 for task in CONFIGS:
  for seed in (42,777,2026):
   source=(SUP if seed==2026 else OLD)/'runs'/task/f'seed{seed}';m=read(source/'metrics.json');assert m['state']=='complete' and m['seed']==seed
   refs.append(dict(task=task,seed=seed,source=str(source),score=100*m['final']['splits']['validation']['accuracy'],metrics=m))
 write(root/'baseline_metrics.json',refs)
 snap=root/'source_snapshot';shutil.copy2(Path(__file__).resolve(),snap/'reproduction'/Path(__file__).name)
 write(root/'launch_manifest.json',dict(created=now(),jobs=JOBS,groups=GROUPS,seeds=[42,777,2026],gpu_policy='GPU2 RTE: one job at a time; GPU3 MRPC: up to two jobs, memory-admitted; existing workers never stopped.',selection='Fixed-final validation accuracy is primary; best-accuracy checkpoints diagnostic only; no early stopping.',source_origin=str(OLD/'source_snapshot'),source_sha256={str(f.relative_to(snap)):hashlib.sha256(f.read_bytes()).hexdigest() for f in snap.rglob('*.py')}))
 write(root/'status.json',dict(state='prepared',total_runs=12,completed=0,active=[],pending=JOBS,failed=[]))
 (root/'README.md').write_text('# RTE/MRPC 正则化独立消融\n\n种子42、777、2026，每任务两组，共12次训练。\n\n|任务|GPU|组A：LoRA dropout|组B：标签平滑|学习率|训练/衰减轮数|\n|---|---:|---|---|---|---|\n|RTE|2|dropout=0.05，LS=0|dropout=0，LS=0.05|2.1e-4|63/63|\n|MRPC|3|dropout=0.10，LS=0|dropout=0.05，LS=0.05|3.5e-4|57/57|\n\n共同：R-Drop alpha=1，batch32，FP32，原WD与warmup，16步校准，755712参数。每组仅更改一个训练正则化参数。标签平滑应用于两个前向的监督CE，KL保持原样；验证损失仍为硬标签CE。主结果为最终轮accuracy，同时保存best_accuracy_adapter与原best_adapter（MRPC旧口径ACC/F1）。\n\nGPU2最多1任务，GPU3最多2任务；完成后自动继续。preflight_validation.json记录数学检查及原脚本/零标签平滑的两轮完全一致性检查。\n')

def preflight(root):
 import torch
 import torch.nn.functional as F
 from types import SimpleNamespace
 spec=importlib.util.spec_from_file_location('regularization_runner',root/'source_snapshot/reproduction/run_roberta_small_glue_regularization.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 x=torch.tensor([[1.2,-.3],[-.2,.9]],dtype=torch.float64,requires_grad=True);y=torch.tensor([[.7,.1],[.3,.4]],dtype=torch.float64,requires_grad=True);labels=torch.tensor([0,1]);a=SimpleNamespace(logits=x,loss=F.cross_entropy(x,labels));b=SimpleNamespace(logits=y,loss=F.cross_entropy(y,labels))
 v,ce,kl=mod.rdrop_loss(a,b,1.,labels,.05)
 def manual(z):
  logp=F.log_softmax(z,dim=-1);return .95*(-logp[range(2),labels]).mean()+.05*(-logp.mean(dim=-1)).mean()
 torch.testing.assert_close(ce,.5*(manual(x)+manual(y)),rtol=0,atol=1e-12);torch.testing.assert_close(v,ce+kl,rtol=0,atol=1e-12)
 v.backward();assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
 v0,ce0,k0=mod.rdrop_loss(a,None,0.,labels,.05);torch.testing.assert_close(v0,manual(x),rtol=0,atol=1e-12);assert k0.item()==0
 def task_smokes(task):
  from safetensors.torch import load_file
  cfg=CONFIGS[task];baseline=dict(task=task,seed=42,scheduler_epochs=cfg['epochs'],label_smoothing=0.,**cfg);results=[]
  cases=[('reference',baseline,True,2),('zero_ls',baseline,False,2)]
  cases += [(g['ablation'],dict(**g,seed=42,scheduler_epochs=g['epochs']),False,1) for g in GROUPS if g['task']==task]
  for label,j,reference,epochs in cases:
   out=root/'smoke'/task/label;log=root/'logs'/f'smoke_{task}_{label}.log'
   free=int(subprocess.check_output(['nvidia-smi','-i',str(j['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>13312,(task,free)
   if not (out/'metrics.json').exists():
    with log.open('x') as f:r=subprocess.run(command(root,j,out,reference,epochs),cwd=root/'source_snapshot',env=env(j['gpu'],42),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    assert r.returncode==0,(task,label,str(log))
   m=read(out/'metrics.json');assert m['state']=='complete' and m['epochs']==epochs and m['adapter_parameters']==755712
   script='reference_small_glue_rdrop.py' if reference else 'run_roberta_small_glue_regularization.py'
   assert read(out/'run_config.json')['source_sha256']==hashlib.sha256((root/'source_snapshot/reproduction'/script).read_bytes()).hexdigest()
   if not reference:assert (out/'best_accuracy_adapter/adapter_model.safetensors').exists() and m['hyperparameters']['label_smoothing']==j['label_smoothing']
   results.append(dict(task=task,case=label,passed=True))
  a=root/'smoke'/task/'reference';b=root/'smoke'/task/'zero_ls'
  ha=[json.loads(x) for x in (a/'history.jsonl').read_text().splitlines()];hb=[json.loads(x) for x in (b/'history.jsonl').read_text().splitlines()]
  def close_tree(x,y):
   if isinstance(x,dict):
    assert x.keys()==y.keys()
    for k in x:close_tree(x[k],y[k])
   elif isinstance(x,float):assert abs(x-y)<=1e-6,(task,x,y)
   else:assert x==y,(task,x,y)
  for x,y in zip(ha,hb):
   close_tree(x,y);assert x['splits']['validation']['accuracy']==y['splits']['validation']['accuracy']
  assert read(a/'allocation.json')['rank_pattern']==read(b/'allocation.json')['rank_pattern']
  wa=load_file(a/'final_adapter/adapter_model.safetensors');wb=load_file(b/'final_adapter/adapter_model.safetensors');assert wa.keys()==wb.keys()
  deltas=[]
  for key in wa:
   if 'lora_A' in key:
    bk=key.replace('lora_A','lora_B');deltas.append((wa[bk].double()@wa[key].double()-wb[bk].double()@wb[key].double()).abs().max().item())
  assert max(deltas)<1e-5,(task,'effective adapter update differs',max(deltas))
  repeat_evidence=None
  if task=='rte':
   hr=[json.loads(x) for x in (root/'smoke/rte/reference_repeat/history.jsonl').read_text().splitlines()]
   for x,y in zip(ha,hr):close_tree(x,y)
   repeat_evidence=dict(original_repeat_max_loss_difference=max(abs(x['train_loss']-y['train_loss']) for x,y in zip(ha,hr)),modified_max_loss_difference=max(abs(x['train_loss']-y['train_loss']) for x,y in zip(ha,hb)))
  return dict(task=task,smokes=results,two_epoch_zero_ls_accuracy_exact=True,history_absolute_tolerance=1e-6,history_bitwise_equal=ha==hb,weights_bitwise_equal=all(torch.equal(wa[k],wb[k]) for k in wa),max_effective_deltaW_difference=max(deltas),original_repeat=repeat_evidence)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(task_smokes,CONFIGS))
 write(root/'preflight_validation.json',dict(passed=True,updated=now(),math_loss_gradient_check=True,tasks=results));print(json.dumps(results),flush=True)

def validate(root,info,ref):
 out=Path(info['output']);m=read(out/'metrics.json');hp=copy.deepcopy(ref['metrics']['hyperparameters']);hp.update(dropout=info['dropout'],label_smoothing=info['label_smoothing'])
 assert m['state']=='complete' and m['task']==info['task'] and m['seed']==info['seed']
 assert m['hyperparameters']==hp,(m['hyperparameters'],hp)
 assert m['epochs']==m['planned_epochs']==m['final']['epoch']==info['epochs'] and m['optimizer_steps']==info['steps']
 assert not m['stopped_early'] and m['selected']==m['final'] and m['adapter_parameters']==755712
 assert m['gpu_count']==1 and m['visible_gpus']==str(info['gpu']) and m['global_batch_size']==32
 h=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()];assert len(h)==info['epochs'] and h[-1]==m['final']
 assert read(out/'allocation.json')['rank_pattern']==read(Path(ref['source'])/'allocation.json')['rank_pattern']
 best=max(h,key=lambda r:r['splits']['validation']['accuracy']);assert m['best_accuracy']==best
 for checkpoint in ['final_adapter','best_adapter','best_accuracy_adapter']:assert (out/checkpoint/'adapter_model.safetensors').exists()
 last=m['final'];assert abs(last['train_loss']-last['train_ce_loss']-last['train_kl_loss'])<1e-5
 score=100*last['splits']['validation']['accuracy'];return dict(**info,score=score,baseline=ref['score'],delta=score-ref['score'],best_accuracy=m['best_accuracy_percent'],best_epoch=best['epoch'],metrics=m)

def summarize(root,completed,active,pending,failed):
 refs=read(root/'baseline_metrics.json');groups=[]
 for config in GROUPS:
  done=[r for r in completed if r['group']==config['group']];bs=[r['score'] for r in refs if r['task']==config['task']]
  g=dict(**config,complete=len(done)==3,baseline_mean=statistics.mean(bs),per_seed=[{k:r[k] for k in ['seed','score','baseline','delta','best_accuracy','best_epoch']} for r in sorted(done,key=lambda r:r['seed'])])
  if len(done)==3:g.update(final_mean=statistics.mean(r['score'] for r in done),final_sample_std=statistics.stdev(r['score'] for r in done),mean_delta=statistics.mean(r['delta'] for r in done))
  groups.append(g)
 write(root/'summary.json',dict(updated=now(),complete=len(completed)==12,seeds=[42,777,2026],groups=groups));write(root/'completed_runs.json',completed)
 write(root/'status.json',dict(state='running' if active else 'failed' if failed else 'waiting_for_gpu' if pending else 'complete',updated=now(),total_runs=12,completed=len(completed),pending=pending,failed=failed,active=[dict(**info,training_status=read(Path(info['output'])/'status.json')) for _,info in active]))
 if len(completed)==12:
  lines=['# 正则化消融最终结果','','seed42/777/2026，固定最终轮accuracy。','','|配置|旧配置均值|本次均值±SD|差值|','|---|---:|---:|---:|']
  for g in groups:lines.append(f"|{g['group']}|{g['baseline_mean']:.4f}|{g['final_mean']:.4f} ± {g['final_sample_std']:.4f}|{g['mean_delta']:+.4f}|")
  (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def controller(root):
 pending=list(JOBS);active=[];completed=[];failed=[];refs={(r['task'],r['seed']):r for r in read(root/'baseline_metrics.json')}
 while pending or active:
  running=[]
  for proc,info in active:
   rc=proc.poll()
   if rc is None:running.append((proc,info));continue
   try:
    if rc:raise RuntimeError(f'Exit {rc}; see {info["log"]}')
    completed.append(validate(root,info,refs[(info['task'],info['seed'])]))
   except Exception as exc:failed.append(dict(**info,error=repr(exc)))
  active=running
  for gpu,limit,reserve in [(2,1,13312),(3,2,8192)]:
   if sum(i['gpu']==gpu for _,i in active)>=limit:continue
   if any(i['gpu']==gpu and read(Path(i['output'])/'status.json').get('state') not in ('running','complete') for _,i in active):continue
   free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if free<reserve:continue
   job=next((j for j in pending if j['gpu']==gpu),None)
   if job is None:continue
   out=root/'runs'/job['group']/f"seed{job['seed']}";log=root/'logs'/f"{job['group']}_seed{job['seed']}.log";cmd=command(root,job,out)
   with log.open('x') as f:proc=subprocess.Popen(cmd,cwd=root/'source_snapshot',env=env(gpu,job['seed']),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   info=dict(**job,pid=proc.pid,output=str(out),log=str(log),command=cmd,started=now());active.append((proc,info));pending.remove(job);print(json.dumps(info),flush=True)
  summarize(root,completed,active,pending,failed)
  if pending or active:time.sleep(10)
 return 1 if failed else 0
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--prepare',action='store_true');parser.add_argument('--preflight',action='store_true');parser.add_argument('--launch',action='store_true');parser.add_argument('--controller',action='store_true');a=parser.parse_args();root=a.root
 if a.prepare:prepare(root)
 if a.preflight:preflight(root)
 if a.controller:
  try:sys.exit(controller(root))
  except Exception as exc:write(root/'controller_error.json',dict(updated=now(),error=repr(exc)));raise
 if a.launch:
  assert read(root/'status.json')['state']=='prepared' and read(root/'preflight_validation.json')['passed']
  for name,h in read(root/'launch_manifest.json')['source_sha256'].items():assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==h,name
  with (root/'controller.log').open('x') as f:proc=subprocess.Popen([sys.executable,'-u',str(root/'source_snapshot/reproduction'/Path(__file__).name),'--root',str(root),'--controller'],cwd=P,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(root/'controller_pid.json',dict(pid=proc.pid));print(json.dumps(dict(root=str(root),pid=proc.pid)))
