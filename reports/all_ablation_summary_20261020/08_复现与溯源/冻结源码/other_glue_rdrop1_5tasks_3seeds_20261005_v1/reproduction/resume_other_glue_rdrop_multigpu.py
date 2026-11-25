#!/usr/bin/env python3
"""Adopt the live GPU0 job and distribute pending jobs across GPU0/2/3."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
import launch_other_glue_rdrop as base
ROOT=Path(__file__).resolve().parents[2]
LIMITS={0:2,2:1,3:1}
def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return dict(state=fields[0],start_ticks=fields[19])
    except (OSError,IndexError):return None
class AdoptedProcess:
    def __init__(self,info):self.info=info
    def poll(self):
        identity=process_identity(self.info['pid'])
        if identity and identity['state']!='Z' and identity['start_ticks']==self.info.get('process_start_ticks'):
            return None
        return 0 if base.read(Path(self.info['output'])/'metrics.json').get('state')=='complete' else 1

def controller():
    lock=(ROOT/'scheduler_multigpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    saved=base.read(ROOT/'scheduler_migration/resume_state.json')
    pending=saved['pending'];completed=saved['completed_runs'];failed=saved['failed']
    active=[(AdoptedProcess(info),info) for info in saved['active']]
    baseline={(r['task'],r['seed']):r for r in base.read(ROOT/'baseline_metrics.json')}
    try:
        while pending or active:
            running=[]
            for proc,info in active:
                code=proc.poll()
                if code is None:running.append((proc,info));continue
                try:
                    if code:raise RuntimeError(f'Training exited unsuccessfully; see {info["log"]}')
                    completed.append(base.validate(info,baseline[(info['task'],info['seed'])]))
                except Exception as exc:failed.append(dict(**info,error=repr(exc)))
            active=running
            if not failed:
                for gpu in (2,3,0,0):
                    if not pending or sum(i['gpu']==gpu for _,i in active)>=LIMITS[gpu]:continue
                    if any(i['gpu']==gpu and base.read(Path(i['output'])/'status.json').get('state') not in ('running','complete') for _,i in active):continue
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    reserve=13312 if pending[0]['task']=='sst2' else 8192
                    if free<reserve+2048:continue
                    job=pending.pop(0);out=ROOT/'runs'/job['task']/f"seed{job['seed']}";log=ROOT/'logs'/f"{job['task']}_seed{job['seed']}.log"
                    assert not out.exists(),f'Refusing to duplicate {out}'
                    cmd=base.command(ROOT,job,out)
                    with log.open('x') as f:
                        proc=subprocess.Popen(cmd,cwd=ROOT/'source_snapshot',env=base.env(gpu,job['seed']),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                    identity=process_identity(proc.pid)
                    info=dict(**job,gpu=gpu,pid=proc.pid,output=str(out),log=str(log),command=cmd,started=base.now(),process_start_ticks=identity['start_ticks'] if identity else None)
                    active.append((proc,info));print(json.dumps(info),flush=True)
            base.summarize(ROOT,completed,active,pending,failed)
            if failed and not active:return 1
            if pending or active:time.sleep(10)
        return 0
    except Exception as exc:
        base.write(ROOT/'controller_error.json',dict(error=repr(exc),updated=base.now(),active=[i for _,i in active],pending=pending));raise
if __name__=='__main__':sys.exit(controller())
