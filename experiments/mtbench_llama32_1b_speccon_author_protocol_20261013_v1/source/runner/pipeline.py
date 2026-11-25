#!/usr/bin/env python3
"""Freeze, train, generate and optionally judge SpecCon with the DPLoRA protocol."""
import argparse
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile

from common import (AUTHOR_COMMIT, AUTHOR_PYTHON, AUTHOR_ROOT, BASE_SHA256,
                    check_environment, read_json, rows, sha256, summarize_judgments,
                    validate_answers, verify_snapshot, write_json)

PROJECT = Path("/workspace/SpecCon-LoRA")
HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = (AUTHOR_ROOT / "experiments/mtbench_llama32_1b_dp06_author_env_gpu2_20261012_v1"
                    / "mtbench_author_env_generation/answers.jsonl")


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(args):
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("Choose a new output directory; existing experiments are never overwritten")
    cfg = read_json(args.config)
    versions = check_environment(AUTHOR_ROOT / "requirements_alpaca.txt")
    model = Path(cfg["model"])
    base_config = read_json(model / "config.json")
    if (base_config["model_type"], base_config["num_hidden_layers"], base_config["hidden_size"]) != ("llama", 16, 2048):
        raise RuntimeError("Expected the local Llama-3.2-1B base checkpoint")
    if sha256(model / "model.safetensors") != BASE_SHA256:
        raise RuntimeError("Local weights differ from the previously verified official 1B base")
    if len(read_json(cfg["dataset"])) != 52002:
        raise RuntimeError("Expected the 52,002-example Alpaca dataset")
    if cfg["speccon"]["scaling_mode"] != "sqrt_rank":
        raise ValueError("This runner preserves SpecCon sqrt-rank scaling")
    cfg["seeds"] = list(dict.fromkeys(cfg["seeds"]))
    if not cfg["seeds"]:
        raise ValueError("At least one seed is required")
    # Export original tracked files from the audited commit, never a modified tree.
    paths = ["FastChat", "data/alpaca.py", "utils/determinism.py", "requirements_alpaca.txt",
             "training/train_alpaca.py", "evaluation/mt_bench.py", "scripts/run_mt_bench.sh"]
    archive = subprocess.check_output(["git", "-C", str(AUTHOR_ROOT), "archive", AUTHOR_COMMIT, *paths])
    output.mkdir(parents=True)
    author = output / "source/author"
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            target = author / member.name
            if not target.resolve().is_relative_to(author.resolve()):
                raise RuntimeError("Invalid archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(member).read())
    # User algorithm snapshot is byte-for-byte the CURRENT local implementation.
    shutil.copytree(PROJECT / "speccon_lora", output / "source/speccon_lora",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runner = output / "source/runner"
    runner.mkdir()
    for path in HERE.glob("*.py"):
        shutil.copy2(path, runner / path.name)
    write_json(output / "config.json", cfg)
    tracked = [output / "config.json"] + sorted(p for p in (output / "source").rglob("*") if p.is_file())
    input_files = [model / name for name in ("model.safetensors", "config.json", "generation_config.json",
                                            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
    input_files.append(Path(cfg["dataset"]))
    inputs = {str(p): sha256(p) for p in input_files if p.is_file()}
    write_json(output / "manifest.json", {
        "created_utc": now(), "author_commit": AUTHOR_COMMIT, "python": sys.version,
        "executable": sys.executable, "packages": versions, "input_sha256": inputs,
        "snapshot_sha256": {str(p.relative_to(output)): sha256(p) for p in tracked},
        "algorithm_source": str(PROJECT / "speccon_lora"),
        "protocol": {"bench": "mt_bench", "questions": 80, "turns": 160,
                     "conversation": "alpaca_dplora_llama", "dtype": "bfloat16",
                     "max_new_tokens": 1024, "choices": 1, "generation_seed_per_question": 0,
                     "judge": "gpt-4", "judge_mode": "single", "judge_parallel": 2},
        "paid_judging": False,
    })
    write_json(output / "status.json", {"status": "prepared", "created_utc": now(), "training_started": False})
    print(json.dumps({"status": "prepared", "output": str(output), "seeds": cfg["seeds"]}, ensure_ascii=False))


def environment(output, gpu=None, seed=0, paid=False):
    env = dict(os.environ)
    author = output / "source/author"
    env.update(PYTHONPATH=str(author / "FastChat") + os.pathsep + str(output / "source"),
               PATH=str(AUTHOR_PYTHON.parent) + os.pathsep + env.get("PATH", ""),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(seed), CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", ACCELERATE_MIXED_PRECISION="no",
               WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SPECCON_CALIBRATION_SHARDED", "SPECCON_CALIBRATION_CACHE"):
        env.pop(key, None)
    if not paid:
        env.pop("OPENAI_API_KEY", None)
    return env


def run_logged(command, cwd, env, log, input_text=None):
    print(f"Running; log: {log}", flush=True)
    write_json(log.with_suffix(".command.json"), {"argv": command, "cwd": str(cwd), "started_utc": now()})
    with log.open("a") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                input=input_text, text=True)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; inspect {log}")


def model_id(output, seed):
    return re.sub(r"[^A-Za-z0-9_-]", "_", output.name) + f"_seed{seed}"


def stage_data(output, seed):
    work = output / f"seed{seed}/mtbench"
    original = output / "source/author/FastChat/fastchat/llm_judge/data"
    for relative in ("judge_prompts.jsonl", "mt_bench/question.jsonl", "mt_bench/reference_answer/gpt-4.jsonl"):
        dest = work / "data" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha256(dest) != sha256(original / relative):
            raise RuntimeError("Evaluation data changed: " + str(dest))
        if not dest.exists():
            shutil.copy2(original / relative, dest)
    return work


def render_answers(work, answers):
    doc = ["# SpecCon MT-Bench 生成文本", "", "80 题、160 轮；此文件不代表 GPT-4 评分。", ""]
    for question in rows(work / "data/mt_bench/question.jsonl"):
        qid = question["question_id"]
        doc += [f"## Q{qid} · {question['category']}", ""]
        for index in range(2):
            doc += [f"### 第 {index + 1} 轮", ""]
            for title, text in (("题目", question["turns"][index]),
                                ("回答", answers[qid]["choices"][0]["turns"][index])):
                fence = "`" * (max([len(s) for s in re.findall(r"`+", text)] + [2]) + 1)
                doc += [f"**{title}**", "", fence + "text", text, fence, ""]
    (work / "ANSWERS.md").write_text("\n".join(doc) + "\n")
    if DEFAULT_BASELINE.exists():
        baseline = {r["question_id"]: r for r in rows(DEFAULT_BASELINE)}
        pairs = [{"question_id": q, "turn": i + 1, "exact_match":
                  answers[q]["choices"][0]["turns"][i] == baseline[q]["choices"][0]["turns"][i],
                  "speccon": answers[q]["choices"][0]["turns"][i],
                  "dplora": baseline[q]["choices"][0]["turns"][i]}
                 for q in sorted(answers) for i in range(2)]
        write_json(work / "text_comparison.json", {"baseline": str(DEFAULT_BASELINE),
                   "baseline_sha256": sha256(DEFAULT_BASELINE), "exact_matches": sum(r["exact_match"] for r in pairs),
                   "turns": 160, "quality_conclusion": None, "pairs": pairs})


def generate(output, seed, gpu):
    run = output / f"seed{seed}"
    completed = read_json(run / "train_complete.json")
    for name, digest in completed["final_files_sha256"].items():
        if sha256(run / "final_model" / name) != digest:
            raise RuntimeError("Final model weights changed")
    work = stage_data(output, seed)
    answer = work / "answers.jsonl"
    questions = work / "data/mt_bench/question.jsonl"
    mid = model_id(output, seed)
    if answer.exists():
        # A failed generation is kept for diagnosis; never append duplicates to it.
        try:
            valid = validate_answers(answer, questions, mid)
        except (ValueError, KeyError, RuntimeError):
            answer.rename(work / ("answers.partial." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl"))
        else:
            render_answers(work, valid)
            complete = work / "generation_complete.json"
            if complete.exists() and read_json(complete)["answers_sha256"] != sha256(answer):
                raise RuntimeError("Previously completed answers changed")
            if not complete.exists():
                write_json(complete, {"questions": 80, "turns": 160, "model_id": mid,
                                     "answers_sha256": sha256(answer), "paid_judging": False,
                                     "completed_utc": now()})
            return
    script = output / "source/author/FastChat/fastchat/llm_judge/gen_model_answer.py"
    command = [str(AUTHOR_PYTHON), "-u", str(script), "--model-path", str(run / "final_model"),
               "--model-id", mid, "--bench-name", "mt_bench", "--conv-template", "alpaca_dplora_llama",
               "--dtype", "bfloat16", "--num-gpus-per-model", "1", "--num-gpus-total", "1",
               "--max-new-token", "1024", "--num-choices", "1", "--answer-file", str(answer)]
    run_logged(command, work, environment(output, gpu, seed), work / "generation.log")
    valid = validate_answers(answer, questions, mid)
    render_answers(work, valid)
    write_json(work / "generation_complete.json", {"questions": 80, "turns": 160, "model_id": mid,
               "answers_sha256": sha256(answer), "paid_judging": False, "completed_utc": now()})


def verify_inputs(output):
    manifest = verify_snapshot(output)
    check_environment(output / "source/author/requirements_alpaca.txt")
    for path, digest in manifest["input_sha256"].items():
        if sha256(path) != digest:
            raise RuntimeError("Frozen input changed: " + path)


def selected_seeds(output, requested):
    configured = read_json(output / "config.json")["seeds"]
    chosen = configured if requested is None else list(dict.fromkeys(requested))
    if not chosen or not set(chosen) <= set(configured):
        raise ValueError("Requested seed is absent from the frozen config")
    return chosen


def run(args):
    output = args.output.resolve()
    verify_inputs(output)
    for seed in selected_seeds(output, args.seeds):
        run_dir = output / f"seed{seed}"
        run_dir.mkdir(exist_ok=True)
        env = environment(output, args.gpu, seed)
        if args.stage in ("all", "train") and not (run_dir / "train_complete.json").exists():
            write_json(output / "status.json", {"status": "training", "seed": seed, "gpu": args.gpu,
                                                "pid": os.getpid(), "updated_utc": now()})
            run_logged([str(AUTHOR_PYTHON), "-u", str(output / "source/runner/train.py"),
                        "--output", str(output), "--seed", str(seed)], output, env, run_dir / "train.log")
        if args.stage in ("all", "generate"):
            write_json(output / "status.json", {"status": "generating", "seed": seed, "gpu": args.gpu,
                                                "pid": os.getpid(), "updated_utc": now()})
            generate(output, seed, args.gpu)
    write_json(output / "status.json", {"status": "requested_stages_completed", "stage": args.stage,
                                        "seeds": selected_seeds(output, args.seeds), "paid_judging": False,
                                        "completed_utc": now()})


def judge(args):
    if not args.allow_paid_judge:
        raise ValueError("Paid GPT-4 calls require explicit --allow-paid-judge")
    output = args.output.resolve()
    verify_inputs(output)
    from dotenv import dotenv_values
    # No authentication request during prepare/train/generate. Never log secrets.
    values = dotenv_values(args.env_file)
    key = os.environ.get("OPENAI_API_KEY") or values.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment or the server .env file")
    for seed in selected_seeds(output, args.seeds):
        work = stage_data(output, seed)
        mid = model_id(output, seed)
        questions = work / "data/mt_bench/question.jsonl"
        answer = work / "answers.jsonl"
        validate_answers(answer, questions, mid)
        if read_json(work / "generation_complete.json")["answers_sha256"] != sha256(answer):
            raise RuntimeError("Answers changed since generation completed")
        answer_dir = work / "data/mt_bench/model_answer"
        answer_dir.mkdir(exist_ok=True)
        dest = answer_dir / f"{mid}.jsonl"
        if dest.exists() and sha256(dest) != sha256(answer):
            raise RuntimeError("Judgment input has changed")
        if not dest.exists():
            shutil.copy2(answer, dest)
        judgment = work / "data/mt_bench/model_judgment/gpt-4_single.jsonl"
        if judgment.exists():
            # Do not blindly reissue 160 paid requests after an interrupted judge job.
            score = summarize_judgments(judgment, questions, mid)
        else:
            env = environment(output, paid=True)
            env["OPENAI_API_KEY"] = key
            for name in ("OPENAI_API_BASE", "OPENAI_ORGANIZATION"):
                value = os.environ.get(name) or values.get(name)
                if value:
                    env[name] = value
            write_json(work / "judge_manifest.json", {
                "model": "gpt-4", "mode": "single", "parallel": 2, "requests": 160,
                "temperature": 0, "max_tokens": 2048, "answer_sha256": sha256(answer),
                "source_commit": AUTHOR_COMMIT, "created_utc": now(),
                "api_base": "configured override" if env.get("OPENAI_API_BASE") else "SDK official default",
            })
            script = output / "source/author/FastChat/fastchat/llm_judge/gen_judgment.py"
            run_logged([str(AUTHOR_PYTHON), "-u", str(script), "--bench-name", "mt_bench",
                        "--judge-model", "gpt-4", "--mode", "single", "--parallel", "2",
                        "--model-list", mid], work, env, work / "judge.log", input_text="\n")
            score = summarize_judgments(judgment, questions, mid)
        write_json(work / "score.json", score)
        print(json.dumps(score, ensure_ascii=False))


def summary(args):
    output = args.output.resolve()
    scores = []
    seeds = selected_seeds(output, args.seeds)
    for seed in seeds:
        path = output / f"seed{seed}/mtbench/score.json"
        if not path.exists():
            raise RuntimeError(f"No completed GPT-4 score for seed {seed}")
        scores.append({"seed": seed, **read_json(path)})
    result = {"seeds": scores, "mean": statistics.mean(s["score"] for s in scores),
              "std_population": statistics.pstdev(s["score"] for s in scores) if len(scores) > 1 else None,
              "std_sample": statistics.stdev(s["score"] for s in scores) if len(scores) > 1 else None,
              "paper_three_seeds_complete": set(seeds) == {42, 777, 2025}}
    write_json(output / "aggregate_scores.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Freeze inputs/source; no training or API calls")
    p.add_argument("--config", type=Path, default=HERE / "config_1b.json")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("run", help="Train and/or generate locally; never call a paid judge")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--stage", choices=["all", "train", "generate"], default="all")
    p.set_defaults(func=run)
    p = sub.add_parser("judge", help="Explicit paid GPT-4 grading, isolated per seed")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--allow-paid-judge", action="store_true")
    p.add_argument("--env-file", type=Path, default=AUTHOR_ROOT / ".env")
    p.set_defaults(func=judge)
    p = sub.add_parser("summary")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int)
    p.set_defaults(func=summary)
    args = parser.parse_args()
    if args.command == "prepare":
        args.func(args)
        return
    output = args.output.resolve()
    verify_snapshot(output)
    # Execute the frozen runner, so later edits cannot alter an existing experiment.
    frozen = output / "source/runner/pipeline.py"
    if Path(__file__).resolve() != frozen:
        os.execv(str(AUTHOR_PYTHON), [str(AUTHOR_PYTHON), str(frozen), *sys.argv[1:]])
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another command is already using this experiment")
        try:
            args.func(args)
        except Exception as exc:
            write_json(output / "status.json", {"status": "failed", "command": args.command,
                                                "error": str(exc), "updated_utc": now()})
            raise


if __name__ == "__main__":
    main()
