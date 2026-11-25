#!/usr/bin/env python3
"""Apply the user's explicit switch to lambda=.25 confirmation runs."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/SpecCon-LoRA/experiments/qnli_qqp_hparam_sweep_20261022_v1')
spec = importlib.util.spec_from_file_location('sweep_controller', ROOT / 'controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def args_for(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return []


def wait_gone(pid, marker, timeout=15):
    deadline = time.monotonic() + timeout
    while marker in args_for(pid):
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Process {pid} did not stop')
        time.sleep(.2)


def main():
    state = controller.read(ROOT / 'status.json')
    manifest = controller.read(ROOT / 'manifest.json')
    assert not (ROOT / 'STOP').exists(), 'Existing STOP file needs review'
    assert not manifest.get('user_override'), 'Override already applied; inspect state first'
    for relative, digest in manifest['source_hashes'].items():
        assert controller.sha(ROOT / relative) == digest, relative

    desired_ids = {
        f'{task}_lr{lr:.6f}_e{epochs}_lambda0.25_s{seed}'
        for task, lr, epochs in [('qnli', .00048, 49), ('qqp', .00044, 10)]
        for seed in [777, 2026]
    }
    preserved = [a for a in state['active'] if a['id'] in desired_ids]
    stopped = [a for a in state['active'] if a['id'] not in desired_ids]
    assert len(preserved) == 2 and all(a['id'].startswith('qnli_') for a in preserved)
    assert len(stopped) == 2 and all(a['id'].startswith('qqp_lr0.000440_e20_lambda1.5_') for a in stopped)
    cancelled_pending = [j for j in manifest['jobs'] if j['id'] in state['pending'] and j['id'] not in desired_ids]
    assert len(cancelled_pending) == 2
    assert all(j['id'].startswith('qnli_lr0.000480_e49_lambda0.5_') for j in cancelled_pending)
    for entry in preserved + stopped:
        assert entry['output'].encode() in args_for(entry['pid']), entry
    for entry in stopped:
        assert os.getpgid(entry['pid']) == entry['pid'], entry
    old_pid = state['controller_pid']
    assert str(ROOT / 'controller.py').encode() in args_for(old_pid)

    added = []
    for seed in [777, 2026]:
        cfg = next(c for c in manifest['candidates'] if c['id'] == 'qqp_lr0.000440_e10_lambda0.25')
        job = controller.job_for(cfg, seed, 'confirm')
        assert job['id'] not in {j['id'] for j in manifest['jobs']}
        assert not Path(job['output']).exists()
        assert Path(job['rng']).is_file()
        added.append(job)

    backup = ROOT / 'interventions' / ('lambda025_' + time.strftime('%Y%m%d_%H%M%S'))
    backup.mkdir(parents=True, exist_ok=False)
    for name in ['manifest.json', 'status.json', 'results.json', 'launch.json', 'controller.py', 'PROTOCOL.md', 'analysis/REPORT.md', 'analysis/candidates.csv']:
        dest = backup / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(__file__, backup / 'apply_override.py')
    request = dict(
        requested_at=controller.now(),
        user_request='停止其他实验，仅运行 QNLI/QQP lambda=0.25 的 seed777 和 seed2026；保留已运行的目标任务进度。',
        preserved=preserved, stopped=stopped,
        cancelled_pending=[j['id'] for j in cancelled_pending],
        added=[j['id'] for j in added], backup=str(backup),
    )
    controller.write(backup / 'request.json', request)
    print(json.dumps(request, ensure_ascii=False, indent=2), flush=True)

    # Stop only the controller; training children have independent sessions.
    os.kill(old_pid, signal.SIGTERM)
    wait_gone(old_pid, str(ROOT / 'controller.py').encode())
    for entry in stopped:
        assert entry['output'].encode() in args_for(entry['pid'])
        os.killpg(entry['pid'], signal.SIGTERM)
    for entry in stopped:
        try:
            wait_gone(entry['pid'], entry['output'].encode())
        except RuntimeError:
            assert entry['output'].encode() in args_for(entry['pid'])
            os.killpg(entry['pid'], signal.SIGKILL)
            wait_gone(entry['pid'], entry['output'].encode())
    for entry in preserved:
        assert entry['output'].encode() in args_for(entry['pid']), 'Target training unexpectedly stopped'

    cancelled_ids = {a['id'] for a in stopped} | {j['id'] for j in cancelled_pending}
    cancelled = []
    for job in manifest['jobs']:
        if job['id'] not in cancelled_ids:
            continue
        record = dict(job=copy.deepcopy(job), state='cancelled_by_user', cancelled_at=controller.now(),
                      reason='User selected lambda=0.25 confirmation only', was_started=job['id'] in {a['id'] for a in stopped})
        status_path = Path(job['output']) / 'status.json'
        if status_path.exists():
            old_status = controller.read(status_path)
            record['training_at_cancellation'] = old_status
            controller.write(status_path, dict(old_status, state='cancelled_by_user', cancelled_at=record['cancelled_at'], cancellation_reason=record['reason']))
        cancelled.append(record)
    manifest['jobs'] = [j for j in manifest['jobs'] if j['id'] not in cancelled_ids] + added
    manifest['cancelled_jobs'] = cancelled
    manifest['original_confirmations'] = copy.deepcopy(manifest['confirmations'])
    manifest['user_override'] = request
    for task in ['qnli', 'qqp']:
        cfg = next(c for c in manifest['candidates'] if c['task'] == task and c['lam'] == .25)
        score = controller.read(ROOT / 'runs' / (cfg['id'] + '_s42') / 'metrics.json')['final_score_percent']
        manifest['confirmations'][task] = dict(selected_at=controller.now(), configs=[cfg['id']],
            seed42_scores=[score], selection='explicit_user_request_lambda0.25', seeds=[42, 777, 2026])
    assert len(manifest['jobs']) == 31
    assert len({j['id'] for j in manifest['jobs']}) == 31
    assert all(j['id'] in desired_ids for j in manifest['jobs'] if j['id'] not in state['completed'])
    controller.write(ROOT / 'cancellations.json', cancelled)
    controller.write(ROOT / 'manifest.json', manifest)
    (ROOT / 'USER_OVERRIDE_20261027.md').write_text(
        '# User-selected lambda=0.25 confirmation\n\n'
        f'Applied: {controller.now()}\n\n'
        'The user superseded the automatic top-two confirmation plan. Only QNLI and QQP at lambda=0.25, seeds 777 and 2026, remain scheduled. '
        'QNLI uses LR 0.00048 and 49 epochs; QQP uses LR 0.00044 and 10 epochs. All other settings, frozen training source, and per-seed reference RNG are unchanged. '
        'The two existing QNLI processes continue without restarting.\n\n'
        'Cancelled: two running QQP lambda=1.5/20-epoch jobs and two queued QNLI lambda=0.5 jobs. '
        'Their descriptors and any partial training artifacts are preserved in cancellations.json and their original directories. '
        'Cancelled jobs are excluded from active planned totals and final-score aggregation. Previously completed results remain available.\n\n'
        f'Prior manifest, reports, launcher and controller are backed up at `{backup}`.\n'
    )
    with (ROOT / 'controller.log').open('a') as log:
        log.write('\n' + controller.now() + ' USER_OVERRIDE lambda=0.25 only; cancelled=' + ','.join(sorted(cancelled_ids)) + '\n')
        log.flush()
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'controller.py'), 'run'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    controller.write(ROOT / 'launch.json', dict(pid=process.pid, started=controller.now(),
        controller=str(ROOT / 'controller.py'), previous_pid=old_pid, intervention=str(backup)))
    print('RESTARTED_CONTROLLER', process.pid, flush=True)
    time.sleep(2)
    assert process.poll() is None, 'Controller exited; inspect controller.log'
    print('PRESERVED_PIDS', [a['pid'] for a in preserved], flush=True)


if __name__ == '__main__':
    main()
