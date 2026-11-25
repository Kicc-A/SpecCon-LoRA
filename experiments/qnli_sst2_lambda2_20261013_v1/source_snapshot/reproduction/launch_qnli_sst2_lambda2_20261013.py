#!/usr/bin/env python3
"""Run only lambda=2.0 using the frozen QNLI/SST-2 sweep protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import launch_qnli_sst2_lambda_sweep_20261012 as sweep

sweep.ROOT = sweep.PROJECT / 'experiments/qnli_sst2_lambda2_20261013_v1'
sweep.LAMBDAS = [2.0]
ROOT = sweep.ROOT


def prepare():
    sweep.prepare()
    source = Path(__file__)
    target = ROOT / 'source_snapshot/reproduction' / source.name
    shutil.copy2(source, target)
    manifest = sweep.read(ROOT / 'manifest.json')
    manifest['source_sha256'][str(target.relative_to(ROOT / 'source_snapshot'))] = (
        hashlib.sha256(target.read_bytes()).hexdigest())
    manifest['change_from_previous_sweep'] = (
        'Previous 2026-10-12 sweep stopped by user; only lambda=2.0 is scheduled here.')
    sweep.write(ROOT / 'manifest.json', manifest)


def launch():
    assert sweep.read(ROOT / 'status.json')['state'] == 'prepared'
    controller = ROOT / 'source_snapshot/reproduction' / Path(__file__).name
    with (ROOT / 'controller.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(controller), '--controller'],
            cwd=ROOT / 'source_snapshot/reproduction', stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    sweep.write(ROOT / 'controller_pid.json', dict(pid=process.pid, started=sweep.now()))
    print(json.dumps(dict(pid=process.pid, root=str(ROOT))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.controller:
        sys.exit(sweep.controller())
    if args.launch:
        launch()
