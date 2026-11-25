#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess,time
R=Path(__file__).resolve().parents[1]
P=json.loads((R/'configs/run_plan.json').read_text())
while True:
 subprocess.run(['/root/anaconda3/bin/python',str(R/'src/finalize.py')],check=True)
 done=sum((Path(j['output'])/'metrics.json').exists() for j in P)
 print(time.strftime('%Y-%m-%d %H:%M:%S'),'done',done,'/12',flush=True)
 if done==12:break
 alive=any(subprocess.run(['tmux','has-session','-t',f'speccon_head_gpu{g}_20261110'],
             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 for g in (0,2,3))
 if not alive:raise RuntimeError(f'All workers ended, only {done}/12 finished')
 time.sleep(60)
subprocess.run(['/root/anaconda3/bin/python',str(R/'src/finalize.py')],check=True)
print('complete and audited',flush=True)
