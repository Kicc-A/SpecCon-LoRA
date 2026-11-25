#!/usr/bin/env python3
"""Frozen experiment matrix with resource-aware scheduling and auditable final results."""
import argparse
import ast
import collections
import copy
from datetime import datetime
import difflib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

PROJECT = Path("/workspace/SpecCon-LoRA")
ROOT = PROJECT / "experiments/extended_ablation_20261021_v2"
PYTHON = "/root/anaconda3/bin/python"
TASKS = ["cola", "mrpc", "rte"]
SEEDS = [42, 777, 2026]
GPUS = [0, 1, 2, 3]
MIN_FREE = {"cola": 6144, "mrpc": 9216, "rte": 14336}
TRAIN_ROWS = {"cola": 8551, "mrpc": 3668, "rte": 2490}
MECHANISM = PROJECT / "experiments/independent_calibration_mechanism_20261019_v1"


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def set_option(command, key, value):
    if key in command:
        command[command.index(key) + 1] = str(value)
    else:
        command.extend([key, str(value)])


def patch_runner(original):
    from launch_main_ablation_gpu12_20261016 import patch_runner as main_patch
    source = main_patch(original)
    source = replace_once(source,
        "from main_ablation_support import prepare_main_ablation",
        "from extended_ablation_support import prepare_extended, read_spec")
    source = replace_once(source, "    args = parser.parse_args()\n",
        '    parser.add_argument("--extension-spec", type=Path, required=True)\n'
        "    args = parser.parse_args()\n")
    source = replace_once(source,
        '    PARAM_BUDGET = 1_327_104 if args.ablation_variant in "AB" else 755_712',
        '    extra_spec = read_spec(args.extension_spec)\n'
        '    PARAM_BUDGET = extra_spec["budget"]')
    source = replace_once(source,
        '    cfg["calibration_blocks"] = 0 if args.ablation_variant == "A" else 4',
        '    cfg["calibration_blocks"] = extra_spec["blocks"]\n'
        '    cfg["extension"] = extra_spec')
    source = replace_once(source,
        '    cfg["calibration_steps"] = 0 if args.ablation_variant == "A" else calibration_steps',
        '    cfg["calibration_steps"] = extra_spec["steps"]')
    source = replace_once(source,
        "    model, report = prepare_main_ablation(model, calibration_loader, loss_fn, speccon,\n"
        "        device, args.ablation_variant, args.allocation_reference, args.expected_allocation, output)",
        "    model, report = prepare_extended(model, calibration_loader, loss_fn, speccon,\n"
        "        device, args.extension_spec, args.allocation_reference, output)")
    source = source.replace("args.rdrop_alpha > 0",
                            '(args.rdrop_alpha > 0 or extra_spec["two_forwards"])')
    source = replace_once(source, '    epochs = args.smoke_epochs if args.smoke else cfg["epochs"]',
        '    if args.smoke:\n'
        '        train_dataset = train_dataset.select(range(min(128, len(train_dataset))))\n'
        '    epochs = args.smoke_epochs if args.smoke else cfg["epochs"]')
    source = source.replace('"MainAblation-" + args.ablation_variant',
                            '"ExtendedAblation-" + extra_spec["name"]')
    ast.parse(source)
    # All loss, evaluation, seeding and data functions remain exactly frozen.
    before = {n.name: ast.dump(n) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    assert all(before[n] == after[n] for n in before if n not in {"parse_args", "main"})
    return source


def base_spec(row, group, name, **overrides):
    spec = dict(group=group, name=name, task=row["task"], seed=row["seed"],
        budget=755712, allocation="adaptive", init="svd", steps=16, blocks=4,
        gamma=1.2, **{"lambda": row["hyperparameters"]["lambda"]},
        two_forwards=True, rdrop_alpha=1.0)
    spec.update(overrides)
    return spec


def prepare():
    if ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite {ROOT}")
    manifest = read(PROJECT / "reproduction/current_glue_manifest.json")
    old_root = PROJECT / "experiments/glue_main_ablation_3seeds_gpu12_20261016_v3"
    old_manifest = read(old_root / "manifest.json")
    def old_source(task, seed, variant):
        cell = next(c for c in old_manifest["cells"] if
                    (c["task"], c["seed"], c["variant"]) == (task, seed, variant))
        if cell["reused"]:
            return Path(next(r for r in old_manifest["references"] if r["id"] == cell["id"])["source"])
        return Path(cell["output"])
    rows = {(r["task"], r["seed"]): r for r in manifest["runs"]
            if r["task"] in TASKS and r["seed"] in SEEDS}
    assert len(rows) == 9
    for row in rows.values():
        assert sha(row["runner"]) == row["runner_sha256"]
        patch_runner(Path(row["runner"]).read_text())
    ROOT.mkdir()
    for directory in ["logs", "specs", "references", "preflight", "source_snapshot", "attempts"]:
        (ROOT / directory).mkdir()
    source_root = ROOT / "source_snapshot"
    for name in [Path(__file__).name, "extended_ablation_support.py",
                 "launch_main_ablation_gpu12_20261016.py"]:
        shutil.copy2(PROJECT / "reproduction" / name, source_root / name)
    shutil.copy2(PROJECT / "reproduction/current_glue_manifest.json", ROOT / "original_manifest.json")
    jobs, reused, smoke = [], [], []
    frozen_rows = {}
    source_hashes = {}
    for (task, seed), row in rows.items():
        ref = ROOT / "references" / task / f"seed{seed}"
        ref.mkdir(parents=True)
        for name in ["allocation.json", "training_rng.pt", "training_rng.json", "metrics.json"]:
            shutil.copy2(Path(row["source"]) / name, ref / name)
        snap = source_root / task / f"seed{seed}"
        shutil.copytree(Path(row["cwd"]) / "speccon_lora", snap / "speccon_lora",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (snap / "reproduction").mkdir()
        original = Path(row["runner"]).read_text()
        patched = patch_runner(original)
        runner = snap / "reproduction" / Path(row["runner"]).name
        runner.write_text(patched)
        shutil.copy2(PROJECT / "reproduction/extended_ablation_support.py", snap / "extended_ablation_support.py")
        (ref / "runner.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(True), patched.splitlines(True),
            fromfile=row["runner"], tofile=str(runner))))
        frozen_rows[task, seed] = dict(row=row, snapshot=snap, runner=runner, reference=ref)
        metric = read(ref / "metrics.json")
        assert metric["state"] == "complete" and metric["epochs"] == row["hyperparameters"]["epochs"]
        reused.append(dict(group="baseline", name="full", task=task, seed=seed,
            source=row["source"], metrics=str(ref / "metrics.json"), budget=755712))

    def make_job(spec, is_smoke=False):
        task, seed = spec["task"], spec["seed"]
        frozen = frozen_rows[task, seed]
        row = frozen["row"]
        ident = f'{task}_{spec["name"]}_s{seed}'
        if is_smoke:
            ident = "smoke_" + ident
        output = ROOT / ("preflight" if is_smoke else "runs") / ident
        spec_path = ROOT / "specs" / (ident + ".json")
        write(spec_path, spec)
        command = list(row["command"])
        command[0], command[2] = PYTHON, str(frozen["runner"])
        for key, value in [("--output-dir", output), ("--ablation-variant", "F"),
                ("--extension-spec", spec_path), ("--rdrop-alpha", spec["rdrop_alpha"]),
                ("--calibration-steps", max(4, spec["steps"])), ("--subspace-consistency-lambda", spec["lambda"]),
                ("--allocation-reference", frozen["reference"] / "allocation.json"),
                ("--training-rng-reference", frozen["reference"] / "training_rng.pt")]:
            set_option(command, key, value)
        if is_smoke:
            command += ["--smoke", "--smoke-epochs", "1"]
        return dict(id=ident, spec=spec, command=command, cwd=str(frozen["snapshot"]),
            output=str(output), log=str(ROOT / "logs" / (ident + ".log")),
            reference=str(frozen["reference"]), smoke=is_smoke,
            expected_hyperparameters=row["hyperparameters"], min_free_mib=MIN_FREE[task])

    # List every hypothesis before seeing any additional validation results.
    for task in TASKS:
        for seed in SEEDS:
            row = rows[task, seed]
            for init in ["random", "svd"]:
                jobs.append(make_job(base_spec(row, "equal_budget", "near_uniform_" + init,
                    allocation="near_uniform", init=init)))
            for init in ["orthogonal_svdnorm", "random_svdnorm", "svd_randomnorm"]:
                jobs.append(make_job(base_spec(row, "initialization", init, init=init,
                    expect_reference_allocation=True)))
            for method in ["speccon", "uniform8"]:
                extra = {}
                if method == "uniform8":
                    old_initial = read(old_source(task, seed, "A") / "initial_state.json")
                    extra = dict(steps=0, blocks=0,
                        expected_initial_adapter_sha256=old_initial["initial_adapter_sha256"])
                jobs.append(make_job(base_spec(row, "rdrop_equal_compute", method + "_two_ce",
                    allocation="adaptive" if method == "speccon" else "uniform",
                    init="svd" if method == "speccon" else "random", uniform_rank=8,
                    budget=755712 if method == "speccon" else 1327104, rdrop_alpha=0.0,
                    expect_reference_allocation=method == "speccon", **extra)))
            for h in [2, 8, 16]:
                jobs.append(make_job(base_spec(row, "blocks", f"H{h}", blocks=h)))
            for n in [128, 256, 1024]:
                jobs.append(make_job(base_spec(row, "calibration_size", f"N{n}", steps=n // 32)))
            for gamma in [0.7, 1.0, 1.5]:
                jobs.append(make_job(base_spec(row, "gamma", f"gamma{gamma}", gamma=gamma)))
            for rank in [3, 6, 8]:
                for allocation in ["uniform", "adaptive"]:
                    if rank == 8 and allocation == "uniform":
                        continue  # Existing B is exactly uniform8 + gradient-SVD.
                    jobs.append(make_job(base_spec(row, "budget_curve", f"budget_r{rank}_{allocation}",
                        budget=165888 * rank, allocation=allocation, uniform_rank=rank)))
    mechanism = read(MECHANISM / "manifest.json")
    assert mechanism["model_seed"] == 42 and mechanism["a_dropout_seed"] == 7001
    for task in TASKS:
        row = rows[task, 42]
        for rep in range(5):
            ids = mechanism["tasks"][task]["replicates"][rep]["orders"][0]["ids"]
            raw = MECHANISM / "runs" / task / f"a{rep}_order0_raw.pt"
            assert raw.is_file()
            for lam in [0.0, row["hyperparameters"]["lambda"]]:
                jobs.append(make_job(base_spec(row, "calibration_subset", f"subset{rep}_lambda{lam}",
                    independent_raw=str(raw), independent_ids=ids, replicate=rep, **{"lambda": lam})))

    # Import existing A/B/D/F cells as provenance-bearing references only.
    for cell in old_manifest["cells"]:
        if cell["variant"] not in ["A", "B", "D", "E"]:
            continue
        source = Path(cell["output"])
        # Reused cells keep their actual source in the separate reference list.
        if cell["reused"]:
            ref = next(r for r in old_manifest["references"] if r["id"] == cell["id"])
            source = Path(ref["source"])
        metric = read(source / "metrics.json")
        assert metric["state"] == "complete" and metric["selected"] == metric["final"]
        destination = ROOT / "references" / cell["task"] / f'seed{cell["seed"]}' / f'old_{cell["variant"]}.json'
        shutil.copy2(source / "metrics.json", destination)
        reused.append(dict(group="baseline", name={"A": "uniform8_random", "B": "uniform8_svd", "D": "lambda0", "E": "adaptive_random"}[cell["variant"]],
            task=cell["task"], seed=cell["seed"], source=str(source), metrics=str(destination),
            budget=metric["adapter_parameters"]))

    # Preflights exercise all new mechanisms and all three untouched task-specific losses.
    for task in TASKS:
        smoke.append(make_job(base_spec(rows[task, 42], "preflight", "full_reference",
            expect_reference_allocation=True), True))
    names = ["near_uniform_random", "near_uniform_svd", "orthogonal_svdnorm",
        "random_svdnorm", "svd_randomnorm", "speccon_two_ce", "uniform8_two_ce",
        "H2", "H8", "H16", "N128", "N256", "N1024", "gamma0.7",
        "budget_r3_adaptive", "budget_r6_uniform", "subset0_lambda0.0", "subset0_lambda1.5"]
    for name in names:
        spec = next(j["spec"] for j in jobs if j["spec"]["name"] == name and j["spec"]["task"] == "cola"
                    and j["spec"]["seed"] == 42)
        smoke.append(make_job(spec, True))
    priorities = {g: i for i, g in enumerate(["equal_budget", "initialization", "rdrop_equal_compute",
        "blocks", "calibration_size", "gamma", "budget_curve", "calibration_subset"])}
    jobs.sort(key=lambda j: (priorities[j["spec"]["group"]], SEEDS.index(j["spec"]["seed"]),
                            j["spec"]["name"], TASKS.index(j["spec"]["task"])))
    assert len(jobs) == 219, len(jobs)
    assert len({j["id"] for j in jobs}) == len(jobs)
    for p in source_root.rglob("*.py"):
        source_hashes[str(p.relative_to(ROOT))] = sha(p)
    for p in (ROOT / "specs").glob("*.json"):
        source_hashes[str(p.relative_to(ROOT))] = sha(p)
    for p in (ROOT / "references").rglob("*"):
        if p.is_file():
            source_hashes[str(p.relative_to(ROOT))] = sha(p)
    write(ROOT / "manifest.json", dict(created=now(), jobs=jobs, preflights=smoke, references=reused,
        source_hashes=source_hashes, python=PYTHON, gpus=GPUS,
        task_seeds=SEEDS, expected_train_rows=TRAIN_ROWS,
        policy="All configurations frozen before training; final epoch; no early stopping or dev selection.",
        groups=dict(collections.Counter(j["spec"]["group"] for j in jobs)),
        independent_calibration_manifest=str(MECHANISM / "manifest.json"),
        independent_calibration_manifest_sha256=sha(MECHANISM / "manifest.json")))
    write(ROOT / "status.json", dict(state="prepared", formal_jobs=len(jobs), preflight_jobs=len(smoke)))
    (ROOT / "PROTOCOL.md").write_text("""# 新增消融：预先冻结方案

CoLA/MRPC/RTE，RoBERTa-base。训练配方、学习率、固定末轮、FP32、冻结分类头、
LoRA scaling=2、batch32、训练样本顺序及训练前CPU/CUDA RNG均与原正式组件实验配对。
普通实验种子42/777/2026；不按新结果挑选配置，不将smoke或引用计为新增训练。

| 实验 | 新训练数 | 对照 |
| --- | ---: | --- |
| 精确等预算 | 18 | 755712参数；attention rank4、FFN rank5，随机/SVD初始化；与F/E配对 |
| 初始化方向/尺度 | 27 | 随机正交同范数、Kaiming同SVD范数、SVD同Kaiming范数；B全零 |
| 等计算R-Drop | 18 | SpecCon及uniform8均双前向CE、alpha0，与已有alpha1及单前向alpha0比较 |
| 分块H | 27 | H=2/8/16；总样本512；默认H4复用；每块样本量随H变化 |
| 校准规模 | 27 | N=128/256/1024，固定H4；N512复用；训练集嵌套前缀 |
| gamma | 27 | 0.7/1.0/1.5；默认1.2复用 |
| 预算曲线 | 45 | 497664/995328/1327104，均匀SVD与自适应SVD；uniform8的9项复用 |
| 独立校准子集下游 | 30 | 三任务各5个A子集；lambda0/完整成对；固定模型及训练seed42 |

总计219个新增正式训练，另21个短预检。预算曲线沿用同一学习率而不单独调参。
精确等预算的near-uniform是结构性近均匀分配，不能称所有模块相同rank；
另三档预算上的uniform是严格相同rank。r_min=2、r_max=16保持不变。
随机正交控制匹配SVD行Gram与范数，用于方向比较；Kaiming同范数只匹配Frobenius，
其行正交结构仍不同。SVD同Kaiming范数保留方向，仅改变幅度。

独立子集直接复用已冻结的A原始梯度与块谱；B仅用于既有机制评价，不参与分配。
这些A/B都是训练集划分，完整下游训练仍使用完整训练集，不称泛化留出测试。
5个A重复可能重叠、只有一个训练seed，统计必须与三训练种子结果分开。
H变化存在梯度累加舍入差异，并且改变谱近似分块的随机数；解释为整体分块敏感性。
所有实验记录实际rank、初始化范数、训练RNG、初始adapter、终轮adapter与逐轮指标。

调度只使用本机GPU0/1/2/3，每卡最多一个本队列任务，按显存阈值和采样利用率准入；
已有负载继续运行，高占用卡等负载降低再加入。所有任务保存退出码，OOM至多自动重试
一次（保留原attempt），其他失败记录并继续独立任务；预检失败则不放行正式矩阵。
可创建本目录STOP文件停止新任务投放；正在训练的任务会正常完成。重启运行器会保留
完成结果，未完成目录移动到attempts，不覆盖记录。结果实时写入results.json和RESULTS.md。
""")
    print(json.dumps(dict(root=str(ROOT), jobs=len(jobs), preflights=len(smoke)), ensure_ascii=False))


def environment(job, gpu):
    env = os.environ.copy()
    for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=job["cwd"], OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", PYTHONHASHSEED=str(job["spec"]["seed"]),
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", WANDB_DISABLED="true")
    return env


def validate(job):
    folder = Path(job["output"])
    metric = read(folder / "metrics.json")
    spec = job["spec"]
    assert metric["state"] == "complete" and metric["final"] == metric["selected"]
    assert metric["task"] == spec["task"] and metric["seed"] == spec["seed"]
    hp = metric["hyperparameters"]
    for key in ["lr", "wd", "clip", "max_len", "warmup", "lr_scheduler", "label_smoothing"]:
        assert hp[key] == job["expected_hyperparameters"][key], (job["id"], key)
    assert hp["extension"] == spec
    assert hp["rdrop_alpha"] == spec["rdrop_alpha"]
    assert hp["training_forwards_per_step"] == 2
    assert hp["calibration_steps"] == spec["steps"] and hp["calibration_blocks"] == spec["blocks"]
    assert hp["gamma"] == spec["gamma"] and hp["lambda"] == spec["lambda"]
    epochs = 1 if job["smoke"] else job["expected_hyperparameters"]["epochs"]
    assert metric["epochs"] == epochs and not metric["stopped_early"]
    assert metric["optimizer_steps"] == epochs * (4 if job["smoke"] else math.ceil(TRAIN_ROWS[spec["task"]] / 32))
    assert metric["adapter_parameters"] == spec["budget"]
    assert metric["global_batch_size"] == 32 and metric["precision"] == "FP32"
    assert read(folder / "allocation.json")["rank_report"]["budget_used"] == spec["budget"]
    rng, ref = read(folder / "training_rng.json"), read(Path(job["reference"]) / "training_rng.json")
    assert all(rng[k] == ref[k] for k in ["cpu_sha256", "cuda_sha256"])
    assert (folder / "final_adapter/adapter_model.safetensors").is_file()
    key = "matthews_correlation" if spec["task"] == "cola" else "accuracy"
    score = 100 * metric["final"]["splits"]["validation"][key]
    assert math.isfinite(score)
    return dict(id=job["id"], task=spec["task"], seed=spec["seed"], group=spec["group"],
        name=spec["name"], budget=spec["budget"], score=score, elapsed_seconds=metric["elapsed_seconds"],
        peak_memory_mib=metric["peak_memory_rank0"] / 1024**2, output=str(folder),
        replicate=spec.get("replicate"), consistency_lambda=spec["lambda"])


def summarize(manifest, complete, failed):
    groups = collections.defaultdict(list)
    for item in complete.values():
        groups[item["group"], item["name"], item["task"]].append(item)
    summaries = []
    for (group, name, task), items in sorted(groups.items()):
        scores = [i["score"] for i in items]
        summaries.append(dict(group=group, name=name, task=task, n=len(items),
            mean=statistics.mean(scores), sample_std=statistics.stdev(scores) if len(scores) > 1 else None))
    refs = []
    for item in manifest["references"]:
        metric = read(item["metrics"])
        key = "matthews_correlation" if item["task"] == "cola" else "accuracy"
        refs.append(dict(**item, score=100 * metric["final"]["splits"]["validation"][key], reused=True))
    write(ROOT / "results.json", dict(updated=now(), completed=len(complete),
        total=len(manifest["jobs"]), failed=failed, runs=list(complete.values()),
        descriptive_summary=summaries, references=refs))
    lines = ["# 新增消融实时结果", "", f"更新：{now()}。完成 {len(complete)}/{len(manifest['jobs'])}；失败 {len(failed)}。",
        "只收录通过末轮、参数预算、训练RNG和配置核验的结果。n不足3的普通实验不得当作完整三种子结论。",
        "", "| 组 | 配置 | 任务 | n | 均值 | 样本标准差 |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for item in summaries:
        sd = "—" if item["sample_std"] is None else f'{item["sample_std"]:.3f}'
        lines.append(f'| {item["group"]} | {item["name"]} | {item["task"]} | {item["n"]} | {item["mean"]:.3f} | {sd} |')
    if failed:
        lines += ["", "失败详情见 failed.json；失败不产生分数。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def gpu_status():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    return {int(v[0]): dict(free=int(v[1]), utilization=int(v[2]))
            for line in raw.splitlines() if (v := line.split(","))}


def verify_sources(manifest):
    for relative, digest in manifest["source_hashes"].items():
        assert sha(ROOT / relative) == digest, f"Frozen artifact changed: {relative}"
    assert sha(manifest["independent_calibration_manifest"]) == manifest["independent_calibration_manifest_sha256"]


def run_stage(manifest, items, stage):
    complete, failed, active = {}, {}, {}
    failure_path = ROOT / f"{stage}_failed.json"
    failure_history = read(failure_path) if failure_path.exists() else {}
    pending = []
    for job in items:
        output = Path(job["output"])
        if (output / "metrics.json").exists():
            try:
                complete[job["id"]] = validate(job)
                continue
            except Exception as exc:
                failure_history[job["id"]] = dict(error=repr(exc), time=now())
        if output.exists() and any(output.iterdir()):
            dest = ROOT / "attempts" / (job["id"] + "_" + str(time.time_ns()))
            output.rename(dest)
        pending.append(job)
    samples = {gpu: collections.deque(maxlen=3) for gpu in GPUS}
    retries = collections.Counter()
    while pending or active:
        for gpu, entry in list(active.items()):
            process, job, handle, started = entry
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[gpu]
            try:
                if code:
                    raise RuntimeError(f"exit_code={code}")
                complete[job["id"]] = validate(job)
                print(now(), "DONE", stage, job["id"], complete[job["id"]]["score"], flush=True)
            except Exception as exc:
                log = Path(job["log"]).read_text(errors="replace")[-16000:]
                failure = dict(error=repr(exc), exit_code=code, log=job["log"], time=now())
                if ("CUDA out of memory" in log or "torch.OutOfMemoryError" in log) and retries[job["id"]] < 1:
                    retries[job["id"]] += 1
                    destination = ROOT / "attempts" / (job["id"] + "_" + str(time.time_ns()))
                    Path(job["output"]).rename(destination)
                    shutil.copy2(job["log"], destination / "process.log")
                    job = copy.deepcopy(job)
                    job["min_free_mib"] = min(22528, job["min_free_mib"] + 4096)
                    pending.append(job)
                    failure_history[job["id"]] = dict(**failure, retry_queued=True, attempt=str(destination))
                else:
                    failed[job["id"]] = failure
                print(now(), "FAIL", stage, job["id"], failure, flush=True)
            write(ROOT / f"{stage}_completed.json", complete)
            write(failure_path, {**failure_history, **failed})
            if stage == "formal":
                write(ROOT / "failed.json", failed)
                summarize(manifest, complete, failed)
        try:
            status = gpu_status()
        except (subprocess.SubprocessError, ValueError) as exc:
            print(now(), "GPU_QUERY_RETRY", repr(exc), flush=True)
            time.sleep(10)
            continue
        for gpu in GPUS:
            samples[gpu].append(status[gpu]["utilization"])
        stop = (ROOT / "STOP").exists()
        for gpu in sorted(GPUS, key=lambda g: (status[g]["utilization"], -status[g]["free"])):
            if stop or (stage == "preflight" and failed) or gpu in active or not pending:
                continue
            # Existing work is left running. A saturated card already meets the
            # user's utilization objective and is admitted when its load drops.
            if len(samples[gpu]) < 3 or statistics.mean(samples[gpu]) > 75:
                continue
            job = next((j for j in pending if status[gpu]["free"] >= j["min_free_mib"]), None)
            if job is None:
                continue
            pending.remove(job)
            handle = Path(job["log"]).open("a")
            handle.write("\n" + now() + f" LAUNCH physical_gpu={gpu}\n")
            handle.flush()
            process = subprocess.Popen(job["command"], cwd=job["cwd"], env=environment(job, gpu),
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[gpu] = (process, job, handle, time.time())
            print(now(), "START", stage, job["id"], "gpu", gpu, "pid", process.pid, flush=True)
        snapshot = dict(state="draining" if stop else "running", stage=stage, updated=now(),
            controller_pid=os.getpid(), active=[dict(gpu=g, id=j["id"], pid=p.pid,
                output=j["output"], log=j["log"], elapsed_seconds=time.time()-t)
                for g, (p, j, h, t) in active.items()],
            pending=[j["id"] for j in pending], completed=list(complete), failed=failed,
            gpu_status=status, formal_total=len(manifest["jobs"]))
        write(ROOT / "status.json", snapshot)
        if stop and not active:
            snapshot["state"] = "stopped"
            write(ROOT / "status.json", snapshot)
            return False
        if stage == "preflight" and failed and not active:
            snapshot["state"] = "preflight_failed"
            write(ROOT / "status.json", snapshot)
            return False
        time.sleep(10)
    if stage == "formal":
        summarize(manifest, complete, failed)
        write(ROOT / "status.json", dict(state="complete" if not failed else "complete_with_failures",
            updated=now(), completed=len(complete), total=len(items), failed=failed, active=[]))
    return not failed


def run():
    lock = (ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(ROOT / "manifest.json")
    verify_sources(manifest)
    # Avoid orphaning or duplicating existing children after a controller restart.
    previous = read(ROOT / "status.json")
    for item in previous.get("active", []):
        proc = Path(f'/proc/{item["pid"]}/cmdline')
        if proc.exists() and item["output"].encode() in proc.read_bytes():
            raise RuntimeError(f"Previous child is still running: {item}")
    if run_stage(manifest, manifest["preflights"], "preflight"):
        summarize(manifest, {}, {})
        run_stage(manifest, manifest["jobs"], "formal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "status"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run()
    else:
        print(json.dumps(read(ROOT / "status.json"), ensure_ascii=False, indent=2))
