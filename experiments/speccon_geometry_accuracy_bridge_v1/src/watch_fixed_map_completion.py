#!/usr/bin/env python3
"""Refresh the audit table until the 30 frozen runs finish or a worker stops."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'configs/fixed_map_completion_plan.json').read_text())
SESSIONS=('speccon_fixed_gpu0_20261109','speccon_fixed_gpu2_20261109','speccon_fixed_gpu3_20261109')
FINAL=ROOT/'src/finalize_fixed_map_completion.py'
while True:
    subprocess.run(['/root/anaconda3/bin/python',str(FINAL)],stdout=subprocess.DEVNULL,check=True)
    done=sum((Path(j['output'])/'metrics.json').exists() for j in PLAN)
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} complete={done}/30',flush=True)
    if done==30: break
    living=[s for s in SESSIONS if subprocess.run(['tmux','has-session','-t',s],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0]
    if not living:
        raise RuntimeError(f'Workers ended with only {done}/30 runs complete')
    time.sleep(60)
subprocess.run(['/root/anaconda3/bin/python',str(FINAL)],check=True)
print('Fixed-map completion and audit finished.',flush=True)
