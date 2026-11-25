#!/usr/bin/env python3
"""Conditional, pre-declared seed777/2026 repeat of the same 12 rank swaps."""
import copy
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path('/workspace/SpecCon-LoRA')
ROOT=PROJECT/'experiments/rank_utility_predictivity_v1'
EXPAND=ROOT/'seed_extension'
SEEDS=(777,2026)
RULE={'consensus_spearman_at_least':0.3,
      'consensus_minus_energy_spearman_at_least':0.1,
      'consensus_top3_actual_mean_delta_greater_than':0.0,
      'policy':'If all criteria hold on all 12 pre-registered seed42 swaps, repeat the fixed reference and all 12 swaps for seeds777/2026. No candidate reselection.'}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def decision(analysis):
    if not analysis.get('complete'):
        return False
    rows={v['method']:v for v in analysis['statistics']}
    energy=rows['energy_only_utility']['spearman_rho']
    consensus=rows['consensus_adjusted_utility']
    return (consensus['spearman_rho']>=RULE['consensus_spearman_at_least']
        and consensus['spearman_rho']-energy>=RULE['consensus_minus_energy_spearman_at_least']
        and consensus['top3_actual_mean_delta']>RULE['consensus_top3_actual_mean_delta_greater_than'])


def prepare():
    if (EXPAND/'manifest.json').exists():
        return
    primary=read(ROOT/'manifest.json')
    assert len(primary['jobs'])==13
    jobs=[]
    for seed in SEEDS:
        for base in primary['jobs']:
            job=copy.deepcopy(base)
            name=job['spec']['name']
            ident=f'mrpc_{name}_s{seed}'
            job['id']=ident
            job['spec']['seed']=seed
            job['output']=str(EXPAND/'runs'/ident)
            job['log']=str(EXPAND/'logs'/f'{ident}.log')
            job['reference']=str(PROJECT/f'experiments/extended_ablation_20261021_v2/references/mrpc/seed{seed}')
            command=job['command']
            for flag,value in (('--seed',seed),('--output-dir',job['output']),
                ('--extension-spec',EXPAND/'specs'/f'{ident}.json'),
                ('--training-rng-reference',Path(job['reference'])/'training_rng.pt')):
                command[command.index(flag)+1]=str(value)
            write(EXPAND/'specs'/f'{ident}.json',job['spec'])
            jobs.append(job)
    assert len(jobs)==26
    write(EXPAND/'manifest.json',{'created':now(),'gpu':1,'training_seeds':list(SEEDS),
        'pre_registration':str(ROOT/'pre_registered_swaps.json'),
        'same_twelve_swap_ids':[j['spec']['name'] for j in primary['jobs'][1:]],
        'model_initialization_seed':42,'jobs':jobs,'decision_rule':RULE,
        'source_snapshot':str(ROOT/'source_snapshot')})
    print(now(),'PREPARED',len(jobs),'extension jobs',flush=True)


def analyze(completed):
    import csv
    from scipy.stats import spearmanr
    registered=read(ROOT/'pre_registered_swaps.json')['selected']
    byid={row['id']:row for row in completed.values()}
    output=[];summary=[]
    for seed in SEEDS:
        baseline=byid.get(f'mrpc_reference_s{seed}')
        pairs=[]
        if baseline:
            for item in registered:
                run=byid.get(f'mrpc_{item["swap_id"]}_s{seed}')
                if run:
                    delta=run['score']-baseline['score']
                    output.append({'training_seed':seed,'swap_id':item['swap_id'],
                        'accuracy_delta':delta,'reference_accuracy':baseline['score'],
                        'swap_accuracy':run['score'],
                        **{name+'_predicted_delta':value for name,value in item['predicted_delta'].items()}})
                    pairs.append((item,delta))
        if len(pairs)==12:
            row={'training_seed':seed}
            for method in ('raw_spectral_energy','energy_only_utility','consensus_adjusted_utility'):
                row[method+'_spearman']=float(spearmanr(
                    [item['predicted_delta'][method] for item,_ in pairs],
                    [delta for _,delta in pairs]).statistic)
            summary.append(row)
    EXPAND.mkdir(parents=True,exist_ok=True)
    with (EXPAND/'repeated_seed_results.csv').open('w',newline='') as stream:
        fields=(['training_seed','swap_id','accuracy_delta','reference_accuracy','swap_accuracy']+
            [name+'_predicted_delta' for name in
             ('raw_spectral_energy','energy_only_utility','consensus_adjusted_utility')])
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(output)
    write(EXPAND/'analysis.json',{'complete_seeds':summary,'pairs_complete':len(output),'planned_pairs':24})


def run():
    sys.path.insert(0,str(PROJECT/'reproduction'))
    from run_rank_utility_predictivity_v1 import free_memory,running_pid,validate,launch
    EXPAND.mkdir(parents=True,exist_ok=True)
    write(EXPAND/'decision_rule.json',RULE)
    with (EXPAND/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if (EXPAND/'STOP').exists():
                write(EXPAND/'status.json',{'updated':now(),'state':'stopped'})
                return
            state=read(ROOT/'status.json')
            if state['state']=='complete' and state['complete']==13:
                break
            if state['state'] in ('failed','stopped'):
                write(EXPAND/'status.json',{'updated':now(),'state':'upstream_failed'})
                return
            write(EXPAND/'status.json',{'updated':now(),'state':'waiting_for_seed42',
                'seed42_complete':state.get('complete',0)})
            time.sleep(60)
        analysis=read(ROOT/'analysis.json')
        permitted=decision(analysis)
        write(EXPAND/'decision.json',{'updated':now(),'rule':RULE,'seed42_analysis':analysis,
            'repeat_fixed_candidates':permitted})
        if not permitted:
            write(EXPAND/'status.json',{'updated':now(),'state':'no_predeclared_signal',
                'reason':'Seed42 full-set correlations/top3 did not satisfy frozen rule'})
            print(now(),'NO EXTENSION: frozen signal rule not met',flush=True)
            return
        prepare()
        jobs=read(EXPAND/'manifest.json')['jobs']
        completed=read(EXPAND/'completed.json') if (EXPAND/'completed.json').exists() else {}
        while len(completed)<len(jobs):
            if (EXPAND/'STOP').exists():
                write(EXPAND/'status.json',{'updated':now(),'state':'stopped',
                    'complete':len(completed),'total':len(jobs)})
                return
            job=next(j for j in jobs if j['id'] not in completed)
            folder=Path(job['output'])
            if (folder/'metrics.json').exists():
                completed[job['id']]=validate(job)
                write(EXPAND/'completed.json',completed)
                analyze(completed)
                continue
            pid=running_pid(job)
            if pid is None:
                if folder.exists() and any(folder.iterdir()):
                    archive=EXPAND/'attempts'/f'{job["id"]}_{int(time.time())}'
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    folder.rename(archive)
                free=free_memory();required=job['min_free_mib']+1024
                if free<required:
                    write(EXPAND/'status.json',{'updated':now(),'state':'waiting_for_memory',
                        'complete':len(completed),'total':len(jobs),'next':job['id'],
                        'free_mib':free,'required_mib':required})
                    time.sleep(30)
                    continue
                pid=launch(job)
                print(now(),'START',job['id'],'gpu 1 pid',pid,flush=True)
            write(EXPAND/'status.json',{'updated':now(),'state':'running',
                'complete':len(completed),'total':len(jobs),
                'active':{'id':job['id'],'pid':pid,'gpu':1}})
            while running_pid(job)==pid:
                time.sleep(30)
            if not (folder/'metrics.json').exists():
                raise RuntimeError(f'Training exited before metrics.json: {job["log"]}')
            completed[job['id']]=validate(job)
            write(EXPAND/'completed.json',completed)
            analyze(completed)
            print(now(),'DONE',job['id'],completed[job['id']]['score'],flush=True)
        write(EXPAND/'status.json',{'updated':now(),'state':'complete',
            'complete':len(completed),'total':len(jobs)})


if __name__=='__main__':
    run()
