#!/usr/bin/env python3
"""Single-GPU CoLA with a fixed training budget and final-epoch model selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from speccon_lora import SpecConLoraConfig, prepare_specconlora_model
from ablation_support import prepare_ablation_model

MODEL_PATH = "/workspace/Models/RoBERTa-base"
DATA_ROOT = Path("/workspace/datasets/glue")
PARAM_BUDGET = 755_712
TASK_KEYS = {
    "cola": ("sentence", None), "rte": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"), "stsb": ("sentence1", "sentence2"),
    "sst2": ("sentence", None), "qnli": ("question", "sentence"),
    "mnli": ("premise", "hypothesis"), "qqp": ("question1", "question2"),
}
TASK_CONFIG = {
    "cola": dict(lr=3.7e-4, epochs=67, wd=0.196, clip=1.177, max_len=320, warmup=0.071),
    "rte": dict(lr=2.1e-4, epochs=63, wd=0.198, clip=0.870, max_len=256, warmup=0.111),
    "mrpc": dict(lr=4.6e-4, epochs=57, wd=0.307, clip=1.498, max_len=128, warmup=0.038),
    "stsb": dict(lr=3.5e-5, epochs=67, wd=0.093, clip=0.670, max_len=128, warmup=0.092),
    "sst2": dict(lr=2.8e-4, epochs=38, wd=0.482, clip=0.932, max_len=256, warmup=0.029),
    "qnli": dict(lr=4.8e-4, epochs=49, wd=0.283, clip=0.450, max_len=128, warmup=0.022),
    "mnli": dict(lr=2.4e-4, epochs=27, wd=0.435, clip=1.573, max_len=128, warmup=0.054),
    "qqp": dict(lr=4.4e-4, epochs=10, wd=0.173, clip=0.652, max_len=128, warmup=0.073),
}


class WeightedDataParallel(nn.DataParallel):
    def forward(self, *inputs, **kwargs):
        batch = kwargs.get("input_ids", inputs[0] if inputs else None)
        result = super().forward(*inputs, **kwargs)
        if result.loss is not None and result.loss.ndim:
            sizes = [chunk.size(0) for chunk in batch.chunk(len(self.device_ids), dim=0)]
            weights = result.loss.new_tensor(sizes)
            result["loss"] = (result.loss * weights).sum() / weights.sum()
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="cola", choices=["cola"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.196)
    parser.add_argument("--lr-scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--calibration-steps", type=int)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0)
    parser.add_argument("--training-rng-reference", type=Path)
    parser.add_argument("--scheduler-epochs", type=int, default=None,
                        help="Fixed LR schedule horizon; defaults to the training epoch budget")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--adapter-method", choices=["speccon", "uniform"], required=True)
    parser.add_argument("--allocation-reference", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.rdrop_alpha) or args.rdrop_alpha < 0:
        parser.error("--rdrop-alpha must be finite and nonnegative")
    if args.scheduler_epochs is not None and args.scheduler_epochs < args.epochs:
        parser.error("--scheduler-epochs must be >= --epochs")
    if args.smoke_epochs < 1:
        parser.error("--smoke-epochs must be positive")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("--learning-rate must be finite and positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.calibration_steps is not None and args.calibration_steps < 4:
        parser.error("--calibration-steps must be at least 4 (four calibration blocks)")
    if args.weight_decay is not None and (not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("--weight-decay must be finite and nonnegative")
    if not 0 <= args.lora_dropout < 1:
        parser.error("--lora-dropout must be in [0, 1)")
    return args


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_data(task, tokenizer, max_len, smoke, smoke_train_samples=128):
    raw = load_from_disk(str(DATA_ROOT / task))
    key1, key2 = TASK_KEYS[task]

    def tokenize(batch):
        texts = (batch[key1],) if key2 is None else (batch[key1], batch[key2])
        result = tokenizer(*texts, truncation=True, max_length=max_len)
        result["labels"] = batch["label"]
        return result

    mapped = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    train = mapped["train"]
    evals = {"matched": mapped["validation_matched"], "mismatched": mapped["validation_mismatched"]} if task == "mnli" else {"validation": mapped["validation"]}
    if smoke:
        train = train.select(range(min(smoke_train_samples, len(train))))
        evals = {name: split.select(range(min(128, len(split)))) for name, split in evals.items()}
    return train, evals


def metric_values(task, predictions, references):
    preds, refs = np.asarray(predictions), np.asarray(references)
    if task == "cola":
        return {"matthews_correlation": float(matthews_corrcoef(refs, preds))}
    if task == "stsb":
        pearson = float(pearsonr(preds, refs)[0]); spearman = float(spearmanr(preds, refs)[0])
        return {"pearson": pearson, "spearmanr": spearman, "corr": (pearson + spearman) / 2.0}
    if task in {"mrpc", "qqp"}:
        acc = float((preds == refs).mean()); f1 = float(f1_score(refs, preds))
        return {"accuracy": acc, "f1": f1, "combined": (acc + f1) / 2.0}
    return {"accuracy": float((preds == refs).mean())}


def evaluate_split(model, loader, device, task):
    model.eval(); predictions = []; references = []; loss_sum = 0.0; examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch); count = batch["input_ids"].size(0)
            loss_sum += float(output.loss.detach()) * count; examples += count
            if task == "stsb": predictions.extend(output.logits.squeeze(-1).detach().cpu().tolist())
            else: predictions.extend(output.logits.argmax(-1).detach().cpu().tolist())
            references.extend(batch["labels"].detach().cpu().tolist())
    return {**metric_values(task, predictions, references), "loss": loss_sum / examples, "examples": examples}


def evaluate(model, loaders, device, task):
    details = {name: evaluate_split(model, loader, device, task) for name, loader in loaders.items()}
    if task == "mnli": score = (details["matched"]["accuracy"] + details["mismatched"]["accuracy"]) / 2.0
    elif task == "cola": score = details["validation"]["matthews_correlation"]
    elif task == "stsb": score = details["validation"]["corr"]
    elif task in {"mrpc", "qqp"}: score = details["validation"]["combined"]
    else: score = details["validation"]["accuracy"]
    return {"combined_score": float(score), "splits": details}


def rdrop_loss(first, second, alpha):
    if second is None:
        if alpha != 0:
            raise ValueError("Positive R-Drop alpha requires two forward passes")
        return first.loss, first.loss, first.loss.new_zeros(())
    logp = torch.nn.functional.log_softmax(first.logits, dim=-1)
    logq = torch.nn.functional.log_softmax(second.logits, dim=-1)
    # Sum over classes, mean over examples, average the two KL directions.
    kl = 0.5 * (torch.nn.functional.kl_div(logp, logq, reduction="batchmean", log_target=True)
                + torch.nn.functional.kl_div(logq, logp, reduction="batchmean", log_target=True))
    ce = 0.5 * (first.loss + second.loss)
    return ce + alpha * kl, ce, kl


def main():
    global PARAM_BUDGET
    args = parse_args(); cfg = dict(TASK_CONFIG[args.task]); output = Path(args.output_dir)
    PARAM_BUDGET = 1_327_104 if args.adapter_method == "uniform" else 755_712
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.learning_rate is not None: cfg["lr"] = args.learning_rate
    if args.epochs is not None: cfg["epochs"] = args.epochs
    if args.weight_decay is not None: cfg["wd"] = args.weight_decay
    cfg["lr_scheduler"] = args.lr_scheduler
    cfg["early_stopping"] = False
    cfg["rdrop_alpha"] = args.rdrop_alpha
    cfg["label_smoothing"] = 0.0
    cfg["rdrop_kl_reduction"] = "class_sum_batch_mean_bidirectional_average"
    cfg["training_forwards_per_step"] = 2 if args.rdrop_alpha > 0 else 1
    cfg["scheduler_epochs"] = args.smoke_epochs if args.smoke else (args.scheduler_epochs or cfg["epochs"])
    calibration_steps = args.calibration_steps if args.calibration_steps is not None else (4 if args.smoke else 16)
    cfg["adapter_method"] = args.adapter_method
    cfg["adapter_parameter_budget"] = PARAM_BUDGET
    cfg["calibration_steps"] = calibration_steps if args.adapter_method == "speccon" else 0
    cfg["calibration_blocks"] = 4 if args.adapter_method == "speccon" else 0
    cfg["training_rng_reference"] = str(args.training_rng_reference) if args.training_rng_reference else None
    atomic_json(output / "run_config.json", {
        "task": args.task, "seed": args.seed, "config": cfg, "lora_dropout": args.lora_dropout,
        "smoke": args.smoke, "scheduler_epochs": cfg["scheduler_epochs"],
        "selection": "fixed final epoch; best checkpoint is diagnostic only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    atomic_json(output / "status.json", {"state": "initializing", "task": args.task, "seed": args.seed})
    seed_everything(args.seed)
    if torch.cuda.device_count() != 1: raise RuntimeError(f"Exactly one visible GPU required, got {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    train_dataset, eval_datasets = build_data(args.task, tokenizer, cfg["max_len"], args.smoke,
                                            smoke_train_samples=max(128, calibration_steps * 32))
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    num_labels = 1 if args.task == "stsb" else (3 if args.task == "mnli" else 2)
    problem_type = "regression" if args.task == "stsb" else "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=num_labels, problem_type=problem_type, local_files_only=True, dtype=torch.float32,
    ).to(device)
    calibration_dataset = train_dataset.select(range(min(len(train_dataset), calibration_steps * 32)))
    calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False, collate_fn=collator)

    def loss_fn(current_model, batch):
        return current_model(**{key: value.to(device) for key, value in batch.items()}).loss

    speccon = SpecConLoraConfig(
        target_modules=["query", "key", "value", "dense"], target_prefix="roberta.encoder.layer",
        base_rank=8, r_min=2, r_max=16, lora_alpha=2, lora_dropout=args.lora_dropout,
        calibration_steps=calibration_steps, calibration_blocks=4, use_subspace_consistency=True,
        subspace_consistency_lambda=1.5, subspace_consistency_floor=0.05,
        spectral_oversample=8, spectral_power_iterations=1, spectral_seed=3407,
        init_method="svd_a_zero_b", init_scale=1.0, init_energy_beta=0.5,
        init_energy_eps=1e-8, init_small_b_scale=1e-4, scaling_mode="constant",
        rank_budget_mode="param", param_budget=PARAM_BUDGET, strict_param_budget=True,
        rank_score_gamma=1.2, attention_rank_prior=1.0,
    )
    model, report = prepare_ablation_model(model, calibration_loader, loss_fn, speccon,
        device, args.adapter_method, args.allocation_reference, output)
    # Training sets each layer's scaling directly. Encode the same alpha/r in
    # PEFT metadata so a normal from_pretrained reproduces training inference.
    peft_config = model.peft_config["default"]
    if peft_config.use_rslora:
        raise RuntimeError("This export path expects ordinary LoRA alpha/r scaling")
    peft_config.lora_alpha = 2 * peft_config.r
    peft_config.alpha_pattern = {name: 2 * rank for name, rank in peft_config.rank_pattern.items()}
    for module in model.modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            assert module.scaling["default"] == 2.0

    allocation = {
        "target_names": report["target_names"], "rank_pattern": report["rank_pattern"],
        "rank_summary": report["rank_summary"], "rank_report": report["rank_report"],
        "params": report["params"], "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(output / "allocation.json", allocation)
    if len(report["target_names"]) != 72 or int(report["rank_report"]["budget_used"]) != PARAM_BUDGET:
        raise RuntimeError(f"Allocation invariant failed: {report['rank_report']}")
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable): raise RuntimeError("Non-LoRA trainable parameter found")
    if sum(param.numel() for _, param in trainable) != PARAM_BUDGET: raise RuntimeError("Trainable parameter budget mismatch")
    lora_a = [param for name, param in trainable if "lora_A" in name]
    lora_b = [param for name, param in trainable if "lora_B" in name]
    if len(lora_a) + len(lora_b) != len(trainable) or not lora_a or not lora_b: raise RuntimeError("LoRA+ grouping failed")

    epochs = args.smoke_epochs if args.smoke else cfg["epochs"]
    steps_per_epoch = math.ceil(len(train_dataset) / 32); total_steps = steps_per_epoch * epochs
    scheduler_steps = steps_per_epoch * cfg["scheduler_epochs"]
    warmup_steps = 1 if args.smoke else int(scheduler_steps * cfg["warmup"])
    lr_b = cfg["lr"]; lr_a = lr_b
    optimizer = torch.optim.AdamW([
        {"params": lora_a, "lr": lr_a, "weight_decay": cfg["wd"]},
        {"params": lora_b, "lr": lr_b, "weight_decay": cfg["wd"]},
    ], betas=(0.9, 0.999))
    schedule_factory = get_linear_schedule_with_warmup if args.lr_scheduler == "linear" else get_cosine_schedule_with_warmup
    scheduler = schedule_factory(optimizer, warmup_steps, scheduler_steps)
    eval_loaders = {name: DataLoader(split, sampler=SequentialSampler(split), batch_size=128, collate_fn=collator, num_workers=0) for name, split in eval_datasets.items()}
    # Reuse only RNG state, never model weights, from the paired 16-batch control.
    # This prevents longer calibration and rank-dependent injection from shifting
    # the random stream used by dropout and loader iterators during training.
    if args.training_rng_reference:
        reference = torch.load(args.training_rng_reference, map_location="cpu", weights_only=True)
        torch.set_rng_state(reference["cpu"])
        torch.cuda.set_rng_state(reference["cuda"], device)
    rng_state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
    rng_tmp = output / "training_rng.pt.tmp"
    torch.save(rng_state, rng_tmp)
    rng_tmp.replace(output / "training_rng.pt")
    atomic_json(output / "training_rng.json", {
        "reference": cfg["training_rng_reference"],
        "cpu_sha256": hashlib.sha256(rng_state["cpu"].numpy().tobytes()).hexdigest(),
        "cuda_sha256": hashlib.sha256(rng_state["cuda"].numpy().tobytes()).hexdigest(),
    })
    history = []; best = None; global_step = 0; started = time.time()
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, generator=generator), batch_size=32, collate_fn=collator, num_workers=0)
        model.train(); loss_sum = 0.0; ce_sum = 0.0; kl_sum = 0.0; seen = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}; optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            second = model(**batch) if args.rdrop_alpha > 0 else None
            loss, ce, kl = rdrop_loss(result, second, args.rdrop_alpha)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
            loss.backward(); grad_norm = torch.nn.utils.clip_grad_norm_([param for _, param in trainable], cfg["clip"], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); count = batch["input_ids"].size(0)
            loss_sum += float(loss.detach()) * count
            ce_sum += float(ce.detach()) * count; kl_sum += float(kl.detach()) * count
            seen += count; global_step += 1
            del result, second, loss, ce, kl
            if global_step % 200 == 0 or args.smoke:
                state = {"state": "running", "task": args.task, "epoch": epoch + 1, "step": global_step, "total_steps": total_steps, "train_loss": loss_sum / seen, "train_ce_loss": ce_sum / seen, "train_kl_loss": kl_sum / seen, "rdrop_alpha": args.rdrop_alpha, "lr_a": scheduler.get_last_lr()[0], "lr_b": scheduler.get_last_lr()[1], "gradient_norm": float(grad_norm), "elapsed_seconds": time.time() - started}
                atomic_json(output / "status.json", state); print("[Train]", json.dumps(state), flush=True)
        metrics = evaluate(model, eval_loaders, device, args.task)
        record = {"epoch": epoch + 1, "step": global_step, "train_loss": loss_sum / seen, "train_ce_loss": ce_sum / seen, "train_kl_loss": kl_sum / seen, "rdrop_alpha": args.rdrop_alpha, **metrics}; history.append(record)
        with (output / "history.jsonl").open("a") as handle: handle.write(json.dumps(record) + "\n")
        print("[Validation]", json.dumps(record), flush=True)
        if best is None or record["combined_score"] > best["combined_score"]:
            best = dict(record); model.save_pretrained(output / "best_adapter"); tokenizer.save_pretrained(output / "best_adapter")
    final = history[-1]
    assert len(history) == epochs and global_step == total_steps
    model.save_pretrained(output / "final_adapter")
    tokenizer.save_pretrained(output / "final_adapter")
    result = {
        "state": "complete", "task": args.task, "seed": args.seed, "model": "RoBERTa-base", "method": "SpecCon-LoRA" if args.adapter_method == "speccon" else "Uniform-LoRA",
        "precision": "FP32", "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": 1, "global_batch_size": 32,
        "epochs": len(history), "planned_epochs": epochs, "stopped_early": False, "optimizer_steps": global_step, "adapter_parameters": PARAM_BUDGET,
        "hyperparameters": {**cfg, "rank_range": [2, 16] if args.adapter_method == "speccon" else [8, 8], "init_method": speccon.init_method,
                            "gamma": 1.2 if args.adapter_method == "speccon" else None,
                            "lambda": 1.5 if args.adapter_method == "speccon" else None, "dropout": args.lora_dropout,
                            "scheduler_epochs": cfg["scheduler_epochs"],
                            "scaling": 2.0, "lora_plus_ratio": 1.0,
                            "lr_a": lr_a, "lr_b": lr_b},
        "selection": "fixed final epoch; best checkpoint is diagnostic only",
        "selected": final, "selected_score_percent": 100 * final["combined_score"],
        "selected_adapter": str(output / "final_adapter"),
        "stop_reason": "fixed_epoch_budget_completed",
        "final": final, "final_score_percent": 100 * final["combined_score"],
        "best": best, "best_score_percent": 100 * best["combined_score"],
        "tail5_score_percent": 100 * sum(row["combined_score"] for row in history[-5:]) / len(history[-5:]),
        "elapsed_seconds": time.time() - started, "peak_memory_rank0": torch.cuda.max_memory_allocated(device),
    }
    atomic_json(output / "metrics.json", result); atomic_json(output / "status.json", result)
    print("[Complete]", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
