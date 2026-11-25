#!/usr/bin/env python3
"""Restart the frozen P0-B QQP runs without overwriting failed attempts."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXP = Path("/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1")
SWEEP = Path("/workspace/SpecCon-LoRA/experiments/qnli_qqp_hparam_sweep_20261022_v1")
RUNNER = SWEEP / "source_snapshot/reproduction/run_roberta_other_glue_rdrop.py"
RNG = SWEEP / "references/rng/qqp"
RUNS = EXP / "runs/qqp_lambda0/restart_20261103"
LOGS = EXP / "logs"
STATUS = EXP / "status_p0b_restart.json"
LOG = EXP / "controller_p0b_restart.log"
SEEDS = (42, 777, 2026)


def note(message):
    line = time.strftime("%Y-%m-%dT%H:%M:%S%z") + " " + message
    with LOG.open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def save(entries):
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               "controller_pid": os.getpid(), "entries": entries}, indent=2))
    tmp.replace(STATUS)


def gpu_rows():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    return [tuple(map(int, line.split(","))) for line in output.splitlines()]


def eligible_gpus(busy):
    candidates = []
    for gpu, free, _ in gpu_rows():
        if gpu in busy or free < 16384:
            continue
        samples = []
        for _ in range(3):
            samples.append(next(util for index, _, util in gpu_rows() if index == gpu))
            time.sleep(2)
        if sum(samples) / 3 <= 75:
            candidates.append((gpu, free))
    return [gpu for gpu, _ in sorted(candidates, key=lambda row: -row[1])]


def pid_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except OSError:
        return False


def launch(seed, gpu):
    output = RUNS / f"seed{seed}"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output: {output}")
    reference = RNG / f"seed{seed}/training_rng.pt"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    command = [sys.executable, "-u", str(RUNNER), "--task", "qqp",
               "--seed", str(seed), "--output-dir", str(output),
               "--learning-rate", "0.00044", "--epochs", "10",
               "--scheduler-epochs", "10", "--weight-decay", "0.173",
               "--lora-dropout", "0.0", "--lr-scheduler", "linear",
               "--calibration-steps", "16", "--rdrop-alpha", "1.0",
               "--subspace-consistency-lambda", "0.0",
               "--training-rng-reference", str(reference)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(seed))
    log_path = LOGS / f"qqp_lambda0_seed{seed}_restart.log"
    with log_path.open("ab") as stream:
        proc = subprocess.Popen(command, cwd=SWEEP / "source_snapshot", env=env,
                                stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    note(f"START seed={seed} gpu={gpu} pid={proc.pid} output={output}")
    return proc, output, log_path


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        entries = json.loads(STATUS.read_text())["entries"]
        if tuple(e["seed"] for e in entries) != SEEDS:
            raise RuntimeError("Existing queue does not match frozen seed list")
    else:
        entries = [{"seed": seed, "state": "queued", "gpu": None, "pid": None,
                    "output": str(RUNS / f"seed{seed}")} for seed in SEEDS]
    active = {}
    for entry in entries:
        if entry["state"] == "running":
            active[entry["seed"]] = (None, Path(entry["output"]),
                                      Path(entry["log"]), entry["pid"])
    failed = False
    note(f"START controller pid={os.getpid()} seeds={SEEDS}")
    while True:
        for entry in entries:
            seed = entry["seed"]
            if seed not in active:
                continue
            proc, output, _, pid = active[seed]
            code = proc.poll() if proc is not None else None
            if proc is None and pid_alive(pid):
                continue
            if proc is not None and code is None:
                continue
            del active[seed]
            metrics_path = output / "metrics.json"
            try:
                metrics = json.loads(metrics_path.read_text())
            except (OSError, json.JSONDecodeError):
                metrics = {}
            if metrics.get("state") == "complete" and metrics.get("adapter_parameters") == 755712 and (code is None or code == 0):
                entry["state"] = "complete"
            else:
                entry["state"] = "failed"
                entry["exit_code"] = code if code is not None else "unknown_after_adoption"
                failed = True
            note(f"EXIT seed={seed} state={entry['state']} code={code}")
        if not failed:
            for entry in entries:
                if entry["state"] != "queued" or len(active) >= 2:
                    continue
                available = eligible_gpus({e["gpu"] for e in entries if e["seed"] in active})
                if not available:
                    break
                try:
                    proc, output, log_path = launch(entry["seed"], available[0])
                except Exception as exc:
                    entry["state"] = "failed"
                    entry["error"] = str(exc)
                    failed = True
                    note(f"LAUNCH FAILED seed={entry['seed']} error={exc}")
                    break
                active[entry["seed"]] = (proc, output, log_path, proc.pid)
                entry.update(state="running", gpu=available[0], pid=proc.pid,
                             log=str(log_path))
        save(entries)
        if not active and (failed or all(e["state"] == "complete" for e in entries)):
            note("STOP failed=" + str(failed))
            return 1 if failed else 0
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
