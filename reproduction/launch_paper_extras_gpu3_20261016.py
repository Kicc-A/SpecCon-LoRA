#!/usr/bin/env python3
"""Complete three-seed R-Drop controls and six rank-location controls on GPU 3."""
import argparse
import ast
import copy
import csv
import difflib
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

PROJECT = Path('/workspace/SpecCon-LoRA')
ROOT = PROJECT / 'experiments/glue_paper_extras_gpu3_20261016_v1'
MAIN = PROJECT / 'experiments/glue_main_ablation_3seeds_gpu12_20261016_v3'
OLD = PROJECT / 'experiments/glue_rdrop_ablation_seed42_gpu3_20261016_v2'
TASKS, SEEDS, GPU = ['mrpc', 'rte', 'cola'], [42, 777, 2026], 3
PERMUTATION_SEEDS = {42: 101, 777: 202, 2026: 303}


def driver():
    path = ROOT / 'legacy_driver.py'
    if not path.exists():
        path = OLD / 'controller.py'
    spec = importlib.util.spec_from_file_location('legacy_rdrop', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.GPU = ROOT, GPU
    return module


L = driver()
read, write, digest, now = L.read, L.write, L.digest, L.now


def patch_runner(original):
    source = L.patch_runner(original)
    source = L.replace_once(source, 'from ablation_support import prepare_ablation_model',
        'from paper_extra_support import prepare_paper_extra_model')
    source = L.replace_once(source, '    args = parser.parse_args()\n',
        '    parser.add_argument("--rank-permutation", type=Path)\n    args = parser.parse_args()\n')
    source = L.replace_once(source, '    cfg["adapter_method"] = args.adapter_method\n',
        '    cfg["adapter_method"] = args.adapter_method\n'
        '    if args.rank_permutation: cfg["rank_permutation"] = str(args.rank_permutation)\n')
    source = L.replace_once(source,
        '    model, report = prepare_ablation_model(model, calibration_loader, loss_fn, speccon,\n'
        '        device, args.adapter_method, args.allocation_reference, output)',
        '    model, report = prepare_paper_extra_model(model, calibration_loader, loss_fn, speccon,\n'
        '        device, args.adapter_method, args.allocation_reference, args.rank_permutation, output)')
    ast.parse(source)
    start, end = '    epochs = args.smoke_epochs if args.smoke else cfg["epochs"]', '    result = {\n'
    assert original.split(start)[1].split(end)[0] == source.split(start)[1].split(end)[0]
    defs = lambda text: {n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    before, after = defs(original), defs(source)
    assert all(before[n] == after[n] for n in before if n not in {'main','parse_args'})
    return source


def expected_hp(row, method, alpha, rng, permutation=None):
    hp = copy.deepcopy(row['hyperparameters'])
    hp.update(adapter_method=method, rdrop_alpha=alpha, training_forwards_per_step=2 if alpha else 1,
        adapter_parameter_budget=1327104 if method=='uniform' else 755712,
        training_rng_reference=str(rng))
    if method == 'uniform':
        hp.update(rank_range=[8,8],init_method='none',gamma=None,calibration_steps=0,calibration_blocks=0)
        hp['lambda'] = None
    if permutation:
        hp['rank_permutation'] = str(permutation)
    return hp


def canonical(hp):
    return {k:v for k,v in hp.items() if k not in {
        'adapter_method','adapter_parameter_budget','ablation_variant','training_rng_reference'}}


def read_reference(cell):
    source = Path(cell['source'])
    if not (source / 'metrics.json').exists():
        return None
    metric = read(source / 'metrics.json')
    assert metric['state']=='complete' and metric['selected']==metric['final']
    assert metric['task']==cell['task'] and metric['seed']==cell['seed']
    assert metric['epochs']==metric['final']['epoch']==cell['expected_hyperparameters']['epochs']
    assert metric['adapter_parameters']==cell['adapter_parameters']
    assert metric['precision']=='FP32' and metric['global_batch_size']==32
    assert canonical(metric['hyperparameters'])==canonical(cell['expected_hyperparameters']),cell['id']
    actual_rng, reference_rng = read(source/'training_rng.json'),read(Path(cell['reference'])/'training_rng.json')
    assert all(actual_rng[k]==reference_rng[k] for k in ['cpu_sha256','cuda_sha256'])
    allocation=read(source/'allocation.json')
    assert allocation['rank_report']['budget_used']==cell['adapter_parameters']
    if cell['adapter_method']=='uniform':
        assert len(allocation['rank_pattern'])==72 and set(allocation['rank_pattern'].values())=={8}
    else:
        assert allocation['rank_pattern']==read(Path(cell['reference'])/'allocation.json')['rank_pattern']
    if cell.get('runner_sha256'):
        assert read(source/'run_config.json')['source_sha256']==cell['runner_sha256']
    assert (source/'final_adapter/adapter_model.safetensors').exists()
    if cell['adapter_method']=='uniform' and cell['rdrop_alpha']==1:
        other=ROOT/'runs'/cell['task']/'uniform_rd0'/f'seed{cell["seed"]}'/'initial_state.json'
        if other.exists():
            left,right=read(other),read(source/'initial_state.json')
            assert all(left[k]==right[k] for k in ['initial_adapter_sha256','base_classifier_sha256'])
    return dict(**{k:cell[k] for k in ['id','task','seed','adapter_method','rdrop_alpha','adapter_parameters']},
        score=L.score(metric),metrics=str(source/'metrics.json'),reused=True)


def make_permutation(reference, seed):
    allocation=read(reference/'allocation.json')
    cfg=read('/workspace/Models/RoBERTa-base/config.json')
    h,f=cfg['hidden_size'],cfg['intermediate_size']
    ranks=allocation['rank_pattern']; shapes={}
    for name in ranks:
        if name.endswith('.intermediate.dense'): shape=[f,h]
        elif name.endswith('.output.dense') and '.attention.' not in name: shape=[h,f]
        else: shape=[h,h]
        shapes[name]=shape
    rng=random.Random(PERMUTATION_SEEDS[seed]); permuted={}; groups=[]
    for shape in sorted(set(tuple(s) for s in shapes.values())):
        names=sorted(n for n in ranks if tuple(shapes[n])==shape)
        values=[ranks[n] for n in names]; shuffled=list(values); rng.shuffle(shuffled)
        assert sorted(values)==sorted(shuffled)
        permuted.update(zip(names,shuffled))
        groups.append(dict(shape=list(shape),modules=len(names),rank_multiset=sorted(values)))
    changed=sum(ranks[n]!=permuted[n] for n in ranks)
    assert changed>0 and sum(permuted[n]*sum(shapes[n]) for n in ranks)==755712
    expected=reference/'permuted_reference'; expected.mkdir()
    modified=copy.deepcopy(allocation); modified['rank_pattern']=permuted
    # Global summary is unchanged because each shape group's rank multiset is unchanged.
    write(expected/'allocation.json',modified)
    shutil.copy2(reference/'training_rng.json',expected/'training_rng.json')
    path=reference/'rank_permutation.json'
    write(path,dict(training_seed=seed,permutation_seed=PERMUTATION_SEEDS[seed],
        original_rank_pattern=ranks,permuted_rank_pattern=permuted,module_shapes=shapes,
        shape_groups=groups,changed_modules=changed,adapter_parameters=755712,
        permuted_allocation_path=str(expected/'allocation.json')))
    return path,expected


def prepare():
    base=read(PROJECT/'reproduction/current_glue_manifest.json')
    records={(r['task'],r['seed']):r for r in base['runs'] if r['task'] in TASKS and r['seed'] in SEEDS}
    assert len(records)==9
    for row in records.values():
        assert digest(row['runner'])==row['runner_sha256']
        patch_runner(Path(row['runner']).read_text())
    ROOT.mkdir(exist_ok=False); (ROOT/'logs').mkdir(); (ROOT/'preflight').mkdir()
    shutil.copy2(OLD/'controller.py',ROOT/'legacy_driver.py')
    shutil.copy2(PROJECT/'reproduction/current_glue_manifest.json',ROOT/'original_manifest.json')
    main=read(MAIN/'manifest.json'); write(ROOT/'main_manifest.json',main)
    cells,jobs=[],[]
    for task in TASKS:
        for seed in SEEDS:
            row=records[task,seed]; source=Path(row['source'])
            reference=ROOT/'references'/task/f'seed{seed}'; reference.mkdir(parents=True)
            for name in ['metrics.json','run_config.json','allocation.json','training_rng.pt','training_rng.json']:
                shutil.copy2(source/name,reference/name)
            snapshot=ROOT/'source_snapshot'/task/f'seed{seed}'
            shutil.copytree(Path(row['cwd'])/'speccon_lora',snapshot/'speccon_lora',
                ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            (snapshot/'reproduction').mkdir()
            runner=snapshot/'reproduction'/Path(row['runner']).name
            original=Path(row['runner']).read_text(); modified=patch_runner(original); runner.write_text(modified)
            (snapshot/'ablation_support.py').write_text(L.SUPPORT)
            shutil.copy2(PROJECT/'reproduction/paper_extra_support.py',snapshot/'paper_extra_support.py')
            shutil.copy2(row['runner'],reference/'original_runner.py')
            (reference/'runner.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),
                modified.splitlines(True),fromfile=row['runner'],tofile=str(runner))))
            for method in ['uniform','speccon']:
                for alpha in [0.0,1.0]:
                    cell=dict(id=f'{task}_{method}_rd{int(alpha)}_s{seed}',task=task,seed=seed,
                        adapter_method=method,rdrop_alpha=alpha,adapter_parameters=1327104 if method=='uniform' else 755712,
                        reference=str(reference),expected_hyperparameters=expected_hp(row,method,alpha,reference/'training_rng.pt'))
                    if alpha==1 and method=='speccon':
                        cell.update(source=str(source),reused=True)
                    elif seed==42:
                        cell.update(source=str(OLD/'runs'/task/f'{method}_rd{int(alpha)}'/'seed42'),reused=True)
                    elif alpha==1:
                        a=next(c for c in main['cells'] if (c['task'],c['seed'],c['variant'])==(task,seed,'A'))
                        cell.update(source=a['output'],reused=True,runner_sha256=a['runner_sha256'],external=True)
                    else:
                        output=ROOT/'runs'/task/f'{method}_rd0'/f'seed{seed}'
                        cell.update(source=str(output),reused=False)
                        jobs.append(build_job(cell,row,runner,snapshot,output,reference))
                    cells.append(cell)
            if task in {'mrpc','rte'}:
                permutation,expected=make_permutation(reference,seed)
                cell=dict(id=f'{task}_rankshuffle_s{seed}',task=task,seed=seed,adapter_method='speccon',
                    rdrop_alpha=1.0,adapter_parameters=755712,reference=str(expected),
                    original_reference=str(reference),rank_permutation=str(permutation),
                    expected_hyperparameters=expected_hp(row,'speccon',1.0,reference/'training_rng.pt',permutation))
                output=ROOT/'runs'/task/'rankshuffle'/f'seed{seed}'
                jobs.append(build_job(cell,row,runner,snapshot,output,reference))
    # Eight small-task R-Drop repeats, six rank-location controls, then four CoLA repeats.
    jobs.sort(key=lambda j:(2 if j['task']=='cola' else 1 if j.get('rank_permutation') else 0,
        SEEDS.index(j['seed']),TASKS.index(j['task']),0 if j['adapter_method']=='speccon' else 1))
    assert len(jobs)==18 and len(cells)==36
    preflight=[]
    for task,method,kind in [('mrpc','speccon','rdrop'),('rte','uniform','rdrop'),('mrpc','speccon','shuffle')]:
        job=copy.deepcopy(next(j for j in jobs if j['task']==task and j['seed']==777
            and j['adapter_method']==method and bool(j.get('rank_permutation'))==(kind=='shuffle')))
        job['id']='smoke_'+job['id']; job['smoke']=True
        job['output']=str(ROOT/'preflight'/job['id']); job['log']=str(ROOT/'logs'/(job['id']+'.log'))
        L.set_option(job['command'],'--output-dir',job['output']); job['command'].append('--smoke')
        preflight.append(job)
    for cell in cells:
        if cell['reused'] and not cell.get('external'): assert read_reference(cell)
    shutil.copy2(__file__,ROOT/'controller.py')
    paths=list((ROOT/'source_snapshot').rglob('*.py'))+[ROOT/'controller.py',ROOT/'legacy_driver.py']
    paths += [p for p in (ROOT/'references').rglob('*') if p.is_file()]
    manifest=dict(created=now(),gpu=GPU,tasks=TASKS,seeds=SEEDS,jobs=jobs,preflight=preflight,cells=cells,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        budget_note='Uniform rank8=1327104; SpecCon and all shuffled controls=755712. Cross-method effects are not budget controlled.',
        protocol='Frozen task recipe, final epoch, paired saved training RNG. R-Drop 0 uses one forward, 1 uses two. Shuffle only within identical (out,in) shapes, then initialize from each destination module gradient. Permutation seeds 101/202/303 paired with training seeds 42/777/2026; joint variation, not a crossed design.',
        scheduling='One additional process on GPU3; existing jobs continue. MRPC/RTE before CoLA. No large datasets. No duplicate main-matrix A/F training.')
    write(ROOT/'manifest.json',manifest)
    (ROOT/'README.md').write_text('''# 论文补充实验：GPU3

新增18次训练：R-Drop α=0 的12次补种子（LoRA/SpecCon × MRPC/RTE/CoLA × 777/2026），
以及同形状模块内 rank 置换6次（MRPC/RTE × 42/777/2026）。
优先8次 MRPC/RTE R-Drop，再6次 rank 置换，最后4次 CoLA。
沿用42/57/63轮、原学习率、FP32/batch32、固定末轮、分类头冻结和配对训练 RNG。
LoRA rank8预算1,327,104，SpecCon预算755,712；rank置换严格保持后者预算和每种形状的rank多重集合。
置换种子101/202/303分别配对训练种子42/777/2026，不从多次置换中挑选结果。
置换后每个模块用自己的梯度重新做SVD初始化，B为零；只改rank放置位置。

R-Drop=1结果复用主消融A/F，主队列未完成的A保持待引用，绝不重复训练。
seed42的R-Drop四格复用已经完成的GPU3实验。36格汇总随外部结果更新。
每个方法内部R-Drop比较预算相同；跨方法比较预算不同，单/双前向计算量也不同。
并发训练耗时不能直接作为独占设备下的效率对照。

results.json/csv：三种子四格表；rank_shuffle_results.json/csv：原始分配与置换分配的配对差值。
status.json：GPU3新增队列；checks.json：配置与置换守恒检查；source_snapshot：冻结代码。
三个短程检查通过后才开始18次正式训练。发生错误停止新增队列并保留日志。
''')
    publish(manifest,jobs,preflight,None,[],[],'prepared')
    print(json.dumps(dict(root=str(ROOT),new_runs=18,rdrop_repeats=12,rank_shuffle=6,gpu=GPU)))


def build_job(cell,row,runner,snapshot,output,reference):
    command=list(row['command']); command[0]=sys.executable; command[2]=str(runner)
    for option,value in [('--output-dir',output),('--adapter-method',cell['adapter_method']),
        ('--rdrop-alpha',cell['rdrop_alpha']),('--training-rng-reference',reference/'training_rng.pt'),
        ('--allocation-reference',reference/'allocation.json')]: L.set_option(command,option,value)
    if cell.get('rank_permutation'): L.set_option(command,'--rank-permutation',cell['rank_permutation'])
    return dict(**cell,command=command,cwd=str(snapshot),output=str(output),
        log=str(ROOT/'logs'/(cell['id']+'.log')),runner_sha256=digest(runner),gpu=GPU,
        min_free_mib=14336 if cell['task']=='rte' else 9216)


CPU_CHECK=r'''
import json,runpy,sys,torch
from pathlib import Path
assert not torch.cuda.is_available()
m=json.loads(Path(sys.argv[1]).read_text()); runner=sys.argv[2]
namespace=runpy.run_path(runner)
jobs=[j for j in m['jobs']+m['preflight'] if j['command'][2]==runner]
for job in jobs:
 sys.argv=[runner]+job['command'][3:]; a=namespace['parse_args'](); hp=job['expected_hyperparameters']
 assert a.seed==job['seed'] and a.task==job['task'] and a.rdrop_alpha==job['rdrop_alpha']
 assert a.epochs==hp['epochs'] and a.scheduler_epochs==hp['scheduler_epochs']
 assert a.learning_rate==hp['lr'] and a.weight_decay==hp['wd'] and a.lora_dropout==hp['dropout']
 assert a.calibration_steps==16 and a.adapter_method==job['adapter_method']
 if a.rank_permutation:
  p=json.loads(a.rank_permutation.read_text()); orig,new=p['original_rank_pattern'],p['permuted_rank_pattern']; shapes=p['module_shapes']
  for shape in set(tuple(s) for s in shapes.values()):
   names=[n for n in orig if tuple(shapes[n])==shape]
   assert sorted(orig[n] for n in names)==sorted(new[n] for n in names)
  assert sum(new[n]*sum(shapes[n]) for n in new)==755712
  assert p['changed_modules']==sum(orig[n]!=new[n] for n in new)>0
print(json.dumps(dict(runner=runner,parsed_jobs=len(jobs),state='passed')))
'''


def verify_sources(manifest):
    for relative,checksum in manifest['source_sha256'].items(): assert digest(ROOT/relative)==checksum,relative


def check():
    manifest=read(ROOT/'manifest.json'); verify_sources(manifest); rows=[]
    for runner in sorted({j['command'][2] for j in manifest['jobs']}):
        r=subprocess.run([sys.executable,'-c',CPU_CHECK,str(ROOT/'manifest.json'),runner],
            cwd=Path(runner).parents[1],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
            capture_output=True,text=True,timeout=120)
        if r.returncode:
            write(ROOT/'checks.json',dict(state='failed',runner=runner,stderr=r.stderr,stdout=r.stdout))
            raise RuntimeError(r.stderr or r.stdout)
        rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
    write(ROOT/'checks.json',dict(state='passed',updated=now(),cpu_checks=rows,gpu_smoke_tests='pending'))
    print(json.dumps(dict(state='passed',runners=len(rows),jobs=sum(r['parsed_jobs'] for r in rows))))


def validate(job):
    L.SEED=job['seed']; row=L.validate(job)
    if job.get('rank_permutation'):
        meta=read(job['rank_permutation']); actual=read(Path(job['output'])/'allocation.json')
        assert actual['rank_pattern']==meta['permuted_rank_pattern']
        initial=read(Path(job['output'])/'initial_state.json')
        assert initial['changed_modules']==meta['changed_modules'] and initial['original_allocation_verified']
    return dict(row,rank_permutation=job.get('rank_permutation'),gpu=GPU)


def aggregate(values):
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
        sample_sd=statistics.stdev(values) if len(values)>1 else None,complete=len(values)==3)


def write_csv(name,rows):
    temp=ROOT/(name+'.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temp.replace(ROOT/name)


def publish(manifest,pending,smoke_pending,active,completed,failed,state,waiting=None):
    active_row=None
    if active:
        process,job=active; status=Path(job['output'])/'status.json'
        active_row=dict(id=job['id'],task=job['task'],seed=job['seed'],gpu=GPU,pid=process.pid,
            smoke=bool(job.get('smoke')),output=job['output'],log=job['log'],
            training_status=read(status) if status.exists() else {})
    rows={r['id']:r for r in completed if not r['smoke']}; cells=[]; missing=[]
    for cell in manifest['cells']:
        row=read_reference(cell) if cell['reused'] else rows.get(cell['id'])
        if cell.get('external') and row is None: missing.append(cell['id'])
        cells.append(dict(task=cell['task'],seed=cell['seed'],adapter_method=cell['adapter_method'],
            rdrop_alpha=cell['rdrop_alpha'],adapter_parameters=cell['adapter_parameters'],
            score=row['score'] if row else None,reused=cell['reused'],metrics=row['metrics'] if row else None))
    groups=[]; effects=[]
    for task in TASKS:
        for method in ['uniform','speccon']:
            for alpha in [0.0,1.0]:
                values=[c['score'] for c in cells if (c['task'],c['adapter_method'],c['rdrop_alpha'])==(task,method,alpha) and c['score'] is not None]
                groups.append(dict(task=task,adapter_method=method,rdrop_alpha=alpha,**aggregate(values)))
            paired=[]
            for seed in SEEDS:
                pair={c['rdrop_alpha']:c['score'] for c in cells if (c['task'],c['seed'],c['adapter_method'])==(task,seed,method)}
                if all(v is not None for v in pair.values()): paired.append(dict(seed=seed,delta=pair[1.0]-pair[0.0]))
            effects.append(dict(task=task,adapter_method=method,paired=paired,**aggregate([r['delta'] for r in paired])))
    write(ROOT/'results.json',dict(updated=now(),metrics={'cola':'MCC x100','mrpc':'Accuracy x100','rte':'Accuracy x100'},
        budget_note=manifest['budget_note'],cells=cells,groups=groups,rdrop_gains=effects))
    write_csv('results.csv',cells)
    shuffled=[]
    for job in manifest['jobs']:
        if not job.get('rank_permutation'): continue
        original=next(c for c in cells if (c['task'],c['seed'],c['adapter_method'],c['rdrop_alpha'])==(job['task'],job['seed'],'speccon',1.0))
        actual=rows.get(job['id']); meta=read(job['rank_permutation'])
        shuffled.append(dict(task=job['task'],seed=job['seed'],permutation_seed=meta['permutation_seed'],
            changed_modules=meta['changed_modules'],adapter_parameters=755712,original_score=original['score'],
            shuffled_score=actual['score'] if actual else None,
            original_minus_shuffled=original['score']-actual['score'] if actual else None,
            metrics=actual['metrics'] if actual else None))
    write(ROOT/'rank_shuffle_results.json',dict(updated=now(),cells=shuffled,
        groups=[dict(task=task,**aggregate([r['original_minus_shuffled'] for r in shuffled if r['task']==task and r['original_minus_shuffled'] is not None])) for task in ['mrpc','rte']],
        note='Training and permutation seeds are paired; variation is joint. Exact same budget within each comparison.'))
    write_csv('rank_shuffle_results.csv',shuffled)
    write(ROOT/'status.json',dict(state=state,updated=now(),controller_pid=os.getpid(),gpu=GPU,
        total_new_runs=18,completed_new_runs=sum(not r['smoke'] for r in completed),
        completed_smoke_tests=sum(r['smoke'] for r in completed),active=active_row,
        pending=[j['id'] for j in pending],smoke_pending=[j['id'] for j in smoke_pending],
        completed=completed,failed=failed,waiting=waiting,external_reference_pending=missing))


def controller():
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=read(ROOT/'manifest.json'); verify_sources(manifest)
        assert read(ROOT/'checks.json')['state']=='passed'
        pending=list(manifest['jobs']); smoke_pending=list(manifest['preflight'])
        active=None; completed=[]; failed=[]
        try:
            while pending or smoke_pending or active:
                if active and active[0].poll() is not None:
                    process,job=active
                    if process.returncode: raise RuntimeError(f'exit {process.returncode}; {job["log"]}')
                    completed.append(validate(job)); active=None
                    print(json.dumps(dict(finished=job['id'],validated=True)),flush=True)
                waiting=None
                if not active and (smoke_pending or pending):
                    queue=smoke_pending if smoke_pending else pending; job=queue[0]
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free>=job['min_free_mib']:
                        assert not Path(job['output']).exists()
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU),PYTHONHASHSEED=str(job['seed']),
                            CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
                            HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                        with Path(job['log']).open('x') as stream:
                            process=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,
                                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                        active=process,job; queue.pop(0)
                        print(json.dumps(dict(started=job['id'],gpu=GPU,pid=process.pid)),flush=True)
                    else: waiting=dict(reason='insufficient_free_memory',free_mib=free,needed_mib=job['min_free_mib'])
                smoke_done=sum(r['smoke'] for r in completed)==len(manifest['preflight'])
                if smoke_done:
                    checks=read(ROOT/'checks.json')
                    if checks['gpu_smoke_tests']=='pending':
                        checks['gpu_smoke_tests']=dict(state='passed',updated=now(),count=3)
                        write(ROOT/'checks.json',checks)
                state='preflight' if smoke_pending or (active and active[1].get('smoke')) else 'running' if active else 'waiting_for_gpu' if pending else 'complete'
                publish(manifest,pending,smoke_pending,active,completed,failed,state,waiting)
                if pending or smoke_pending or active: time.sleep(10)
            # A/F reference cells may still be running in the main matrix. Keep
            # refreshing their table without occupying a GPU or launching copies.
            while read(ROOT/'status.json')['external_reference_pending']:
                time.sleep(30)
                publish(manifest,[],[],None,completed,failed,'waiting_for_reference_results')
            publish(manifest,[],[],None,completed,failed,'complete')
        except Exception as error:
            failed.append(dict(error=repr(error),id=active[1]['id'] if active else None))
            publish(manifest,pending,smoke_pending,active,completed,failed,'failed')
            raise


def launch():
    assert read(ROOT/'status.json')['state']=='prepared' and read(ROOT/'checks.json')['state']=='passed'
    with (ROOT/'controller.log').open('x') as stream:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'controller.py'),'--controller'],cwd=PROJECT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'controller_pid.json',dict(pid=p.pid,started=now()))
    print(json.dumps(dict(pid=p.pid,gpu=GPU,new_runs=18,root=str(ROOT))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    for name in ['prepare','check','launch','controller']: group.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.check: check()
    elif args.launch: launch()
    else: controller()
