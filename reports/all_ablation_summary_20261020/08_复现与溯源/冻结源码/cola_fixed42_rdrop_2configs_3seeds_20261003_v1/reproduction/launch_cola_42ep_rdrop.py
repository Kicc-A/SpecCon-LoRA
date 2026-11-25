#!/usr/bin/env python3
"""Two training-only R-Drop candidates with a fixed 42-epoch schedule."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

PROJECT = Path('/workspace/SpecCon-LoRA')
REFERENCE = PROJECT/'experiments/cola_A_fixed42_sched42_3seeds_20261002_v1'
GROUPS = [('RD05', 0.5), ('RD10', 1.0)]
JOBS = [dict(group=group, rdrop_alpha=ls, seed=seed, gpu=2 if seed in (42,2026) else 3)
        for group,ls in GROUPS for seed in (42,777,2026)]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def now():
    return datetime.now().astimezone().isoformat()


def prepare(root):
    assert root.parent == PROJECT/'experiments'
    source = REFERENCE/'source_snapshot'
    for rel, expected in read(REFERENCE/'launch_manifest.json')['source_sha256'].items():
        assert hashlib.sha256((source/rel).read_bytes()).hexdigest() == expected, rel
    root.mkdir(exist_ok=False)
    (root/'logs').mkdir()
    snap = root/'source_snapshot'
    (snap/'reproduction').mkdir(parents=True)
    shutil.copytree(source/'speccon_lora', snap/'speccon_lora', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(PROJECT/'reproduction/run_roberta_cola_rdrop.py', snap/'reproduction')
    shutil.copy2(Path(__file__), snap/'reproduction')
    baseline = []
    for seed in (42,777,2026):
        p = REFERENCE/'runs'/f'seed{seed}'
        m = read(p/'metrics.json')
        assert m['state']=='complete' and m['epochs']==42 and m['hyperparameters']['scheduler_epochs']==42
        baseline.append(dict(seed=seed,source=str(p),score=m['final_score_percent'],metrics=m))
    write(root/'baseline_epoch42.json', baseline)
    write(root/'launch_manifest.json', dict(created=now(), jobs=JOBS, epochs=42,
        scheduler_epochs=42, lr=0.00019, weight_decay=0.196, dropout=0,
        rdrop_alpha=[0.5,1.0], label_smoothing=0.0, rdrop_kl_reduction="class_sum_batch_mean_bidirectional_average", early_stopping=False, resume=False, source_origin=str(source),
        comparison='Training uses mean CE of two independent dropout forwards plus alpha times symmetric KL averaged per example. Calibration and evaluation unchanged. Compared at equal optimizer steps, not equal compute.',
        baseline_origin=str(REFERENCE),
        source_sha256={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in snap.rglob('*.py')}))
    (root/'README.md').write_text("# CoLA 42/42 R-Drop 对照\n\n"
        "RD05=0.5、RD10=1.0，各跑seed42/777/2026；训练和线性衰减均42轮，无早停。\n"
        "正式训练进行两次独立dropout前向，CE取平均，双向KL按类别求和、按样本平均；标签平滑关闭；校准、rank分配、验证CE和MCC保持原协议。\n"
        "LR=1.9e-4，WD=0.196，LoRA dropout=0，warmup=0.071，参数预算755712。\n"
        "无R-Drop基线复用已完成42/42结果：62.928310 ± 1.006550。最终固定第42轮计分。\n"
        "train_loss记录两次CE均值加alpha乘双向KL，train_ce_loss记录两次普通CE均值，train_kl_loss记录双向KL；验证loss始终为普通CE。\n"
        "status.json记录进度；summary.json、results_per_seed.csv和RESULTS.md自动汇总结果。\n")
    write(root/'status.json', dict(state='prepared', total_runs=6, pending=JOBS, active=[], completed=0, failed=[]))


def spawn(root, job):
    out = root/'runs'/job['group']/f"seed{job['seed']}"
    log = root/'logs'/f"{job['group']}_seed{job['seed']}.log"
    cmd = [sys.executable, '-u', str(root/'source_snapshot/reproduction/run_roberta_cola_rdrop.py'),
           '--task', 'cola', '--output-dir', str(out), '--seed', str(job['seed']),
           '--epochs', '42', '--scheduler-epochs', '42', '--learning-rate', '0.00019',
           '--lora-dropout', '0', '--weight-decay', '0.196', '--lr-scheduler', 'linear',
           '--calibration-steps', '16', '--rdrop-alpha', str(job['rdrop_alpha'])]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), PYTHONHASHSEED=str(job['seed']),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with log.open('x') as handle:
        process = subprocess.Popen(cmd, cwd=root/'source_snapshot', env=env,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
    info = dict(**job, pid=process.pid, output=str(out), log=str(log), command=cmd, started=now())
    print(json.dumps(info), flush=True)
    return process, info


def validate(info, baseline):
    out = Path(info['output'])
    m = read(out/'metrics.json')
    hp = m['hyperparameters']
    assert m['state'] == 'complete' and m['seed'] == info['seed'] and m['task'] == 'cola'
    assert m['epochs'] == m['planned_epochs'] == m['final']['epoch'] == 42
    assert m['optimizer_steps'] == 11256 and hp['scheduler_epochs'] == 42
    assert not m['stopped_early'] and hp['early_stopping'] is False
    assert m['selected'] == m['final'] and m['selected_score_percent'] == m['final_score_percent']
    assert hp['lr'] == hp['lr_a'] == hp['lr_b'] == 0.00019
    assert hp['dropout'] == 0 and hp['wd'] == 0.196
    assert hp['rdrop_alpha'] == info['rdrop_alpha']
    assert hp['label_smoothing'] == 0 and hp['training_forwards_per_step'] == 2
    assert hp['rdrop_kl_reduction'] == 'class_sum_batch_mean_bidirectional_average'
    assert m['final']['train_kl_loss'] >= -1e-6
    assert abs(m['final']['train_loss'] - m['final']['train_ce_loss'] - info['rdrop_alpha']*m['final']['train_kl_loss']) < 1e-5
    assert 'train_ce_loss' in m['final']
    assert hp['gamma'] == 1.2 and hp['lambda'] == 1.5 and hp['rank_range'] == [2,16]
    assert hp['calibration_steps'] == 16 and hp['scaling'] == 2 and hp['lora_plus_ratio'] == 1
    assert m['adapter_parameters'] == 755712 and m['global_batch_size'] == 32
    assert m['visible_gpus'] == str(info['gpu']) and m['gpu_count'] == 1
    history = [json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in history] == list(range(1,43)) and history[-1] == m['final']
    rank = read(out/'allocation.json')['rank_pattern']
    assert rank == read(Path(baseline['source'])/'allocation.json')['rank_pattern']
    assert (out/'final_adapter/adapter_model.safetensors').exists()
    cfg = read(out/'final_adapter/adapter_config.json')
    assert len(cfg['rank_pattern']) == 72 and not cfg['use_rslora']
    assert all(cfg['alpha_pattern'][k] == 2*r for k,r in cfg['rank_pattern'].items())
    refhist = [json.loads(line) for line in (Path(baseline['source'])/'history.jsonl').read_text().splitlines()][:42]
    differences = [100*(a['combined_score']-b['combined_score']) for a,b in zip(history,refhist)]
    return dict(**info, score=m['final_score_percent'], reference_score=baseline['score'],
        delta=m['final_score_percent']-baseline['score'],
        max_abs_history_score_delta=max(map(abs,differences)), metrics=m)


def summarize(root, completed, active, pending, failed):
    baseline = read(root/'baseline_epoch42.json')
    bm = statistics.mean(b['score'] for b in baseline)
    bs = statistics.stdev(b['score'] for b in baseline)
    rows = []
    groups = []
    for group,ls in GROUPS:
        done = sorted([r for r in completed if r['group']==group],key=lambda r:r['seed'])
        per_seed = [dict(seed=r['seed'],gpu=r['gpu'],score=r['score'],reference_score=r['reference_score'],
            delta=r['delta'],train_loss=r['metrics']['final']['train_loss'],
            train_ce_loss=r['metrics']['final']['train_ce_loss'],
            train_kl_loss=r['metrics']['final']['train_kl_loss'],
            val_loss=r['metrics']['final']['splits']['validation']['loss']) for r in done]
        item = dict(group=group,rdrop_alpha=ls,complete=len(done)==3,per_seed=per_seed)
        if item['complete']:
            item.update(final_mean=statistics.mean(r['score'] for r in done),
                final_sample_std=statistics.stdev(r['score'] for r in done),
                mean_delta=statistics.mean(r['delta'] for r in done),
                final_val_loss_mean=statistics.mean(r['val_loss'] for r in per_seed))
        groups.append(item)
        rows.extend(dict(group=group,rdrop_alpha=ls,**r) for r in per_seed)
    result = dict(updated=now(),complete=len(completed)==6,epochs=42,scheduler_epochs=42,
        primary_metric='fixed epoch-42 final validation MCC x100',
        baseline_mean=bm,baseline_sample_std=bs,baseline_reused=True,groups=groups)
    write(root/'summary.json',result)
    write(root/'completed_runs.json',completed)
    progress = [dict(**info,training_status=read(Path(info['output'])/'status.json')) for _,info in active]
    state = 'running' if active else 'failed' if failed else 'waiting_for_gpu_memory' if pending else 'complete'
    write(root/'status.json',dict(state=state,updated=now(),total_runs=6,active=progress,
        completed=len(completed),pending=pending,failed=failed))
    if rows:
        with (root/'results_per_seed.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if result['complete']:
        lines=['# CoLA 42/42 R-Drop结果','','三个seed固定最终第42轮；MCC×100，均值±样本SD。','',
            '| 配置 | 最终MCC | 相对无R-Drop基线 |','|---|---:|---:|',
            f'| 无R-Drop（复用） | {bm:.6f} ± {bs:.6f} | 0 |']
        for g in groups:
            lines.append(f"| R-Drop alpha={g['rdrop_alpha']} | {g['final_mean']:.6f} ± {g['final_sample_std']:.6f} | {g['mean_delta']:+.6f} |")
        lines+=['','| 配置 | Seed | 最终MCC | 配对差异 | 验证普通CE |','|---|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f"| {r['group']} | {r['seed']} | {r['score']:.6f} | {r['delta']:+.6f} | {r['val_loss']:.6f} |")
        lines+=['','仅正式训练损失使用R-Drop；校准和验证保持普通CE，rank分配与配对基线一致。']
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def controller(root):
    pending, active, completed, failed = list(JOBS), [], [], []
    baseline = {b['seed']:b for b in read(root/'baseline_epoch42.json')}
    try:
        while pending or active:
            running = []
            for process, info in active:
                code = process.poll()
                if code is None:
                    running.append((process,info)); continue
                try:
                    if code: raise RuntimeError(f'Exit code {code}; see {info["log"]}')
                    completed.append(validate(info, baseline[info['seed']]))
                except Exception as exc:
                    failed.append(dict(**info, error=repr(exc)))
            active = running
            if not failed:
                for gpu in (2,3):
                    if any(i['gpu']==gpu and read(Path(i['output'])/'status.json').get('state') not in ('running','complete') for _,i in active): continue
                    free = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    slots = max(0,(free-4096)//6144)
                    for job in [j for j in pending if j['gpu']==gpu][:slots]:
                        active.append(spawn(root,job)); pending.remove(job)
            summarize(root, completed, active, pending, failed)
            if failed and not active: return 1
            if pending or active: time.sleep(10)
        return 0
    except Exception as exc:
        write(root/'controller_error.json',dict(error=repr(exc),updated=now(),active=[i for _,i in active],pending=pending))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--launch-prepared', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    if args.controller: sys.exit(controller(root))
    if not args.launch_prepared: prepare(root)
    if not args.prepare_only:
        assert read(root/'status.json')['state'] == 'prepared'
        with (root/'controller.log').open('x') as f:
            process = subprocess.Popen([sys.executable,'-u',str(root/'source_snapshot/reproduction'/Path(__file__).name),
                '--root',str(root),'--controller'],cwd=PROJECT,stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'controller_pid.json',dict(pid=process.pid))
        print(json.dumps(dict(root=str(root),pid=process.pid)))
