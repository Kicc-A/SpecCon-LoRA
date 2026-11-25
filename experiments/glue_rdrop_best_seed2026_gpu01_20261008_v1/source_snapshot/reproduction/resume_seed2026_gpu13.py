#!/usr/bin/env python3
"""Adopt existing seed2026 workers and dispatch pending tasks on GPUs1/3."""
import argparse, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
BASE=Path(__file__).with_name('launch_glue_rdrop_seed2026_gpu01.py')
spec=importlib.util.spec_from_file_location('frozen_queue',BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
class Adopted:
    def __init__(self,pid):
        self.pid=pid
        self.identity=self.stat()
    def stat(self):
        try:
            fields=Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()
            return fields[0],fields[19]
        except FileNotFoundError:return None
    def poll(self):
        current=self.stat()
        if current and self.identity and current[1]==self.identity[1] and current[0]!='Z':return None
        return 0  # Completion is checked strictly using the run's final artifacts.
def controller(root):
    state=base.read(root/'migration_gpu13/resume_state.json')
    pending=state['pending'];completed=base.read(root/'completed_runs.json') or [];failed=state['failed'];active=[]
    for record in state['active']:
        info={k:v for k,v in record.items() if k!='training_status'}
        active.append((Adopted(info['pid']),info))
    while pending or active:
        running=[]
        for proc,info in active:
            rc=proc.poll()
            if rc is None:running.append((proc,info));continue
            try:
                if rc:raise RuntimeError(f'Exit {rc}; see {info["log"]}')
                completed.append(base.validate(root,info))
            except Exception as exc:failed.append(dict(**info,error=repr(exc)))
        active=running
        for gpu in (1,3):
            if len(base.processes(gpu))>=2:continue
            if any(i['gpu']==gpu and base.read(Path(i['output'])/'status.json').get('state') not in ('running','complete') for _,i in active):continue
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            job=next((j for j in pending if j['admission_free_mib']<=free),None)
            if job is None:continue
            out=root/'runs'/job['task']/'seed2026';log=root/'logs'/f"{job['task']}_seed2026.log";cmd=base.command(root,job,out)
            with log.open('x') as handle:proc=subprocess.Popen(cmd,cwd=root/'source_snapshot',env=base.env(gpu),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            info=dict(**job,gpu=gpu,pid=proc.pid,output=str(out),log=str(log),command=cmd,started=base.now());active.append((proc,info));pending.remove(job);print(json.dumps(info),flush=True)
        base.summarize(root,completed,active,pending,failed)
        if pending or active:time.sleep(10)
    return 1 if failed else 0
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);args=parser.parse_args()
    try:sys.exit(controller(args.root))
    except Exception as exc:base.write(args.root/'controller_error_gpu13.json',dict(error=repr(exc),updated=base.now()));raise
