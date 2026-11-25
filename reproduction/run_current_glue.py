"""Replay the retained GLUE runs using their original frozen training code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

MANIFEST = Path(__file__).with_name('current_glue_manifest.json')


def build_command(run, output, gpu):
    runner = Path(run['runner'])
    if hashlib.sha256(runner.read_bytes()).hexdigest() != run['runner_sha256']:
        raise RuntimeError(f'Training source changed: {runner}')
    command = list(run['command'])
    command[command.index('--output-dir') + 1] = str(output)
    if '--training-rng-reference' in command:
        reference = Path(command[command.index('--training-rng-reference') + 1])
        if not reference.is_file():
            raise FileNotFoundError(reference)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       PYTHONHASHSEED=str(run['seed']), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return command, environment


def main():
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=sorted({r['task'] for r in manifest['runs']}))
    parser.add_argument('--seeds', type=int, nargs='+', choices=manifest['seeds'], default=manifest['seeds'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output-root', type=Path, required=True, help='New directory; existing results are never overwritten.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without loading models or starting training.')
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    runs = [r for r in manifest['runs'] if r['task'] == args.task and r['seed'] in args.seeds]
    jobs = []
    protected_sources = [Path(r['source']).resolve() for r in manifest['runs']]
    for run in runs:
        output = output_root / run['task'] / f"seed{run['seed']}"
        if any(output == p or p in output.parents or output in p.parents for p in protected_sources):
            parser.error(f'Output overlaps a retained experiment: {output}')
        if output.exists():
            parser.error(f'Output already exists: {output}')
        command, environment = build_command(run, output, args.gpu)
        jobs.append((run, output, command, environment))
    for run, output, command, environment in jobs:
        print(f"CUDA_VISIBLE_DEVICES={args.gpu} {shlex.join(command)}", flush=True)
        if not args.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.with_suffix('.log')
            with log.open('x') as stream:
                subprocess.run(command, cwd=run['cwd'], env=environment,
                               stdin=subprocess.DEVNULL, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
