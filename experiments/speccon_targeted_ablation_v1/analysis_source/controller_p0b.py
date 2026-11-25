#!/usr/bin/env python3
"""P0-B controller: QQP lambda=0 x seeds 42/777/2026 queue.

Launches the frozen sweep runner (source_snapshot of qnli_qqp_hparam_sweep)
with lambda=0 into this experiment's run dirs. Admission per PROTOCOL:
free memory >= 8 GiB and 3-sample mean utilization <= 75%; max 2 concurrent
runs, one per GPU. Never terminates other processes; only waits.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXP = "/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1"
SWEEP = "/workspace/SpecCon-LoRA/experiments/qnli_qqp_hparam_sweep_20261022_v1"
RUNNER = f"{SWEEP}/source_snapshot/reproduction/run_roberta_other_glue_rdrop.py"
CWD = f"{SWEEP}/source_snapshot"
RNG = f"{SWEEP}/references/rng/qqp"
LOGS = f"{EXP}/logs"
RUNS = f"{EXP}/runs/qqp_lambda0"
STATUS = f"{EXP}/status_p0b.json"
CONTROLLER_LOG = f"{EXP}/controller_p0b.log"
PID_FILE = f"{EXP}/controller_p0b_pid.json"

MIN_FREE_MIB = 8 * 1024
MAX_UTIL = 75.0
MAX_CONCURRENT = 2
RNG_SEEDS = ["42", "777", "2026"]


def log(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%S%z") + " " + msg
    with open(CONTROLLER_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_state(seed):
    metrics = Path(RUNS) / f"seed{seed}" / "metrics.json"
    if not metrics.exists():
        return "pending"
    try:
        with open(metrics) as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "pending"
    if m.get("state") == "complete":
        if int(m.get("adapter_parameters", 0)) == 755712:
            return "complete"
        return "bad_params"
    return "running"


def gpu_table():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True)
    rows = []
    for line in out.strip().splitlines():
        idx, free, util = [v.strip() for v in line.split(",")]
        rows.append({"gpu": int(idx), "free_mib": int(free), "util": float(util)})
    return rows


def sample_util(gpu, n=3, gap=2.0):
    utils = []
    for _ in range(n):
        for r in gpu_table():
            if r["gpu"] == gpu:
                utils.append(r["util"])
                break
        time.sleep(gap)
    return sum(utils) / len(utils)


def admitted_gpus(busy):
    admitted = []
    for r in gpu_table():
        if r["gpu"] in busy:
            continue
        if r["free_mib"] < MIN_FREE_MIB:
            continue
        if sample_util(r["gpu"]) > MAX_UTIL:
            continue
        admitted.append((r["gpu"], r["free_mib"]))
    return sorted(admitted, key=lambda x: -x[1])


def launch(seed, gpu):
    out_dir = Path(RUNS) / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", RUNNER,
        "--task", "qqp",
        "--seed", seed,
        "--output-dir", str(out_dir),
        "--learning-rate", "0.00044",
        "--epochs", "10",
        "--scheduler-epochs", "10",
        "--weight-decay", "0.173",
        "--lora-dropout", "0.0",
        "--lr-scheduler", "linear",
        "--calibration-steps", "16",
        "--rdrop-alpha", "1.0",
        "--subspace-consistency-lambda", "0.0",
        "--training-rng-reference", f"{RNG}/seed{seed}/training_rng.pt",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=seed)
    log_path = Path(LOGS) / f"qqp_lambda0_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as stream:
        proc = subprocess.Popen(command, cwd=CWD, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    log(f"START qqp_lambda0_s{seed} gpu {gpu} pid {proc.pid}")
    return proc.pid


def write_status(entries):
    with open(STATUS, "w") as f:
        json.dump({"updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "entries": entries}, f, indent=1)


def main():
    Path(LOGS).mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        json.dump({"controller_pid": os.getpid()}, f)
    entries = [{"seed": s, "state": run_state(s), "gpu": None, "pid": None} for s in RNG_SEEDS]
    active = {e["seed"]: (e["gpu"], e["pid"]) for e in entries if e["state"] == "running"}
    log(f"START controller pid {os.getpid()} queue {RNG_SEEDS}")
    while True:
        # refresh states
        for e in entries:
            prev = e["state"]
            e["state"] = run_state(e["seed"])
            if prev == "running" and e["state"] != "running" and e["seed"] in active:
                del active[e["seed"]]
                log(f"DONE {e['seed']} state={e['state']}")
        pending = [e for e in entries if e["state"] == "pending"]
        if not pending and not active:
            log("ALL COMPLETE")
            write_status(entries)
            return
        bad = [e for e in entries if e["state"] in ("bad_params",)]
        if bad:
            log("FAIL bad_params: " + ",".join(e["seed"] for e in bad))
            write_status(entries)
            return
        while pending and len(active) < MAX_CONCURRENT:
            adm = admitted_gpus({g for g, _ in active.values() if g is not None})
            if not adm:
                break
            gpu, free = adm[0]
            e = pending.pop(0)
            pid = launch(e["seed"], gpu)
            e["gpu"], e["pid"], e["state"] = gpu, pid, "running"
            active[e["seed"]] = (gpu, pid)
            log(f"QUEUE remaining {len(pending)}")
        write_status(entries)
        time.sleep(60)


if __name__ == "__main__":
    main()
