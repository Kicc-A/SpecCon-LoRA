"""Explicit, immutable profiles. Only GPUs 0 and 1 are used."""
import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = '/root/anaconda3/envs/gora/bin/python'
MODEL = '/workspace/Models/Llama-3.1-8B-Base'
TRAIN = '/workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_train100k.input_output.jsonl'
DEV = '/workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_eval10k.input_output.jsonl'
PROFILES = {
    'author': {'gradient_steps': 32, 'gamma': .05, 'algorithm': 'unmodified upstream GoRA'},
    'paper': {'gradient_steps': 64, 'gamma': .08, 'algorithm': 'explicit paper-formula implementation; see paper_changes.py'},
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=PROFILES, default='author')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true', help='Real calibration, then 2 optimizer updates; separate output')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cfg = PROFILES[a.profile]
    name = f'{a.profile}-seed{a.seed}' + ('-smoke' if a.smoke else '')
    out = HERE / ('validation_runs' if a.smoke else 'runs') / name
    args = [
        '--huggingface', '--model-name-or-path', MODEL,
        '--train-dataset-name', 'metamathqa', '--eval-dataset-name', 'gsm8k',
        '--train-dataset-path', TRAIN, '--eval-dataset-path', DEV,
        '--dataset-class-name', 'iterable', '--mode', 'sft',
        '--output-path', str(out.parent), '--experiment-name', name,
        '--use-lora', '--use-lora-plus', '--lora-plus-scaler', '16', '--use-gora',
        '--gradient-est-n-steps', str(cfg['gradient_steps']), '--gora-init-method', 'compress',
        '--gora-max-rank', '32', '--gora-min-rank', '4', '--gora-scale-by-lr',
        '--gora-lr', str(cfg['gamma']), '--gora-rank-stablize', '--gora-dynamic-scaling',
        '--gora-importance-type', 'union_mean', '--lora-rank', '8', '--lora-scaler', '16',
        '--replace-modules', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
        '--run-lora-in-fp32', '--weight-a-init-method', 'kaiming',
        '--batch-size-per-gpu', '1', '--eval-batch-size-per-gpu', '1',
        '--gradient-accumulation-steps', '32', '--bf16', '--activation-checkpoint',
        '--max-len', '1024', '--max-src-len', '1024', '--eval-max-len', '1024', '--eval-max-src-len', '1024',
        '--seed', str(a.seed), '--dataset-seed', '42', '--zero-stage', '2', '--lr', '5e-5',
        '--warmup', '.03', '--auto-warmup-steps', '10', '--auto-warmup-rate', '.05',
        '--betas', '.9', '.999', '--weight-decay', '5e-4', '--lr-decay-style', 'cosine', '--lr-decay-ratio', '.1',
        '--atten-type', 'flash', '--fuse-linear-loss', '--all-reduce-loss', '--save-trainable', '--skip-eval',
        '--diy-optimizer', '--ds-config-path', str(HERE / 'source/ds_config/zero2_config.json'),
    ]
    args += ['--train-iters', '64'] if a.smoke else ['--epochs', '1']
    if a.profile == 'paper':
        args += ['--gora-features-func', 'sqrt']
    cmd = [PYTHON, '-m', 'deepspeed.launcher.runner', '--include', 'localhost:0,1',
           '--master_port', '16781', str(HERE / 'strict_entry.py'), *args]
    record = dict(profile=a.profile, seed=a.seed, smoke=a.smoke, global_batch=64,
                  gpus=[0, 1], source_commit='4037d4d6ba67ff88de87f90b943ff4e3a3649b67',
                  hardware_adaptation='2 GPUs x microbatch1 x accumulation32; upstream shell is 8 x 4 x 2',
                  model_path=MODEL, runtime_model='author-supported HF entry; not original custom llama entry',
                  configuration=cfg, command=cmd)
    scripts = ['launch.py','strict_entry.py','paper_changes.py','verify_export.py','tokenizer_assets.py','preflight.py','evaluate.py']
    record['reproduction_file_sha256'] = {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in scripts}
    print(json.dumps(record, indent=2), flush=True)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return
    env = os.environ.copy()
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(PYTHONPATH=str(HERE / 'source'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
               GORA_PROFILE=a.profile, GORA_RUN_DIR=str(out), OMP_NUM_THREADS='2',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PATH'] = str(Path(PYTHON).parent) + os.pathsep + env.get('PATH','')
    # Train/eval/validation share the same lock, so this package cannot double-book 0/1.
    with (HERE / 'gpu01.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([PYTHON, str(HERE / 'preflight.py'), '--require-free-gpus'], check=True, env=env)
        out.mkdir(parents=True, exist_ok=False)
        evidence = out / 'invocation_code'
        evidence.mkdir()
        for n in scripts:
            (evidence/n).write_bytes((HERE/n).read_bytes())
        (out / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
        with (out / 'console.log').open('w') as log:
            rc = subprocess.run(cmd, cwd=HERE / 'source', env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        # Upstream trainer catches exceptions: exit status alone is insufficient.
        success = (out / 'training_complete.json').exists() and (out / 'final.ckpt').exists()
        if rc or not success:
            print((out / 'console.log').read_text()[-9000:], file=sys.stderr)
            raise SystemExit(rc or 1)
        assert record['reproduction_file_sha256'] == {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in scripts}, 'Reproduction scripts changed during run'
        print(f'Training completed: {out}', flush=True)
        subprocess.run([PYTHON, str(HERE / 'verify_export.py'), '--run', str(out)], check=True, env=env)

if __name__ == '__main__':
    main()
