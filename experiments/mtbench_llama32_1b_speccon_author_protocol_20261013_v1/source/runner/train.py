"""Single-GPU SpecCon Alpaca training; author data/AdamW/accumulation conventions.

Run through pipeline.py so source, dependencies, inputs and configuration are audited.
The original speccon_lora package is copied unchanged into each experiment snapshot.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, get_scheduler

from common import check_environment, read_json, sha256, verify_snapshot, write_json


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AlpacaCollator:
    # Same padding/masking as author's training.train_alpaca._AlpacaCollator.
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, instances):
        inputs = pad_sequence([torch.tensor(i["input_ids"], dtype=torch.long) for i in instances],
                              batch_first=True, padding_value=self.pad_id)
        labels = pad_sequence([torch.tensor(i["labels"], dtype=torch.long) for i in instances],
                              batch_first=True, padding_value=-100)
        return {"input_ids": inputs, "labels": labels, "attention_mask": inputs.ne(self.pad_id).long()}


def optimizer_for(model, config):
    groups = []
    for is_b in (False, True):
        parameters = [p for name, p in model.named_parameters()
                      if p.requires_grad and ("lora_B" in name) == is_b]
        if parameters:
            groups.append({"params": parameters, "lr": config["learning_rate"] *
                           (config["lora_plus_ratio"] if is_b else 1),
                           "weight_decay": config["weight_decay"]})
    # Same HF AdamW defaults as author (including eps=1e-6 and correct_bias=True).
    return AdamW(groups, lr=config["learning_rate"], no_deprecation_warning=True)


@contextmanager
def nonquantized_peft(model):
    """Avoid eager optional bnb import in PEFT 0.14; use its normal Linear backend.

    requirements pins bnb 0.42 + Triton 3.2, whose optional triton.ops import fails.
    Only these two dispatch probes are temporarily disabled; no packages or SpecCon
    source are patched. This context explicitly rejects quantized models.
    """
    if any(getattr(model, flag, False) for flag in ("is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized")):
        raise ValueError("Quantized models are outside this BF16 author protocol")
    backend = importlib.import_module("peft.tuners.lora.model")
    old_8, old_4 = backend.is_bnb_available, backend.is_bnb_4bit_available
    backend.is_bnb_available = lambda: False
    backend.is_bnb_4bit_available = lambda: False
    try:
        yield
    finally:
        backend.is_bnb_available, backend.is_bnb_4bit_available = old_8, old_4


def author_backward(accelerator, loss, accumulation):
    # Preserve the author's actual code, including both divisions: its loop divides
    # by accumulation, and Accelerate 1.4.0.backward divides once more internally.
    # Do not silently substitute a standard single-division accumulation loop.
    accelerator.backward(loss / accumulation)


def evaluate(model, loader, device, micro_batch_size):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for logical in loader:
            # Slice AFTER logical-batch padding, preserving the author's mask/positions.
            for start in range(0, len(logical["input_ids"]), micro_batch_size):
                batch = {k: v[start:start + micro_batch_size].to(device) for k, v in logical.items()}
                n = int((batch["labels"][..., 1:] != -100).sum())
                if n == 0:
                    continue
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss with supervised tokens")
                total += float(loss) * n
                count += n
    if count == 0:
        raise RuntimeError("Validation has no supervised tokens")
    return {"eval_loss": total / count, "perplexity": math.exp(total / count), "valid_tokens": count}


def atomic_save(state, path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(output, seed):
    manifest = verify_snapshot(output)
    source = output / "source"
    author = source / "author"
    check_environment(author / "requirements_alpaca.txt")
    cfg = read_json(output / "config.json")
    if cfg["backward_mode"] != "author_accelerate":
        raise ValueError("This runner preserves the author's accumulation implementation")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one GPU per seed; effective batch is 2 x 8 = 16")
    if not torch.cuda.is_available():
        raise RuntimeError("Full 1B training requires a CUDA GPU")
    run = output / f"seed{seed}"
    run.mkdir(exist_ok=True)
    sys.path.insert(0, str(source))
    from speccon_lora import SpecConLoraConfig, calibrate_specconlora, prepare_specconlora_model
    preprocess = load_module("author_alpaca", author / "data/alpaca.py").preprocess_dataset
    determinism = load_module("author_determinism", author / "utils/determinism.py")
    determinism.set_deterministic_environment(seed)
    accelerator = Accelerator(gradient_accumulation_steps=cfg["gradient_accumulation_steps"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True, use_fast=True)
    pad = "<|finetune_right_pad_id|>"
    if pad not in tokenizer.get_vocab():
        raise RuntimeError("Expected the verified Llama-3.2 tokenizer")
    tokenizer.pad_token = pad
    tokenizer.padding_side = "right"
    raw = read_json(cfg["dataset"])
    if len(raw) != 52002:
        raise RuntimeError("Expected the original 52,002 Alpaca examples")
    raw_dataset = DatasetDict(train=Dataset.from_list(raw))
    train_data, eval_data = preprocess(tokenizer, raw_dataset, cfg["max_length"],
                                      max_eval_samples=cfg["max_eval_samples"], seed=seed,
                                      eval_split_ratio=cfg["eval_split_ratio"])
    collator = AlpacaCollator(tokenizer)
    generator = torch.Generator().manual_seed(seed)
    loader_args = {"collate_fn": collator, "worker_init_fn": determinism.seed_worker,
                   "generator": generator, "num_workers": 0}
    train_loader = DataLoader(train_data, shuffle=True, batch_size=cfg["batch_size"], **loader_args)
    eval_loader = DataLoader(eval_data, batch_size=cfg["eval_logical_batch_size"], **loader_args)
    gs = SpecConLoraConfig(**cfg["speccon"])
    n_calibration = gs.calibration_steps * cfg["calibration_batch_size"]
    calibration_loader = DataLoader(train_data.select(range(n_calibration)), shuffle=False,
                                   batch_size=cfg["calibration_batch_size"], collate_fn=collator)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16,
                                                local_files_only=True).to(accelerator.device)
    base_params = sum(p.numel() for p in model.parameters())
    if base_params != 1235814400 or model.config.model_type != "llama":
        raise RuntimeError(f"Wrong base model: {base_params} parameters")
    model.config.pad_token_id = tokenizer.pad_token_id
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    def loss_fn(current_model, batch):
        loss = current_model(**{k: v.to(accelerator.device) for k, v in batch.items()}).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite calibration loss; allocation would be invalid")
        return loss

    cache_path = run / "calibration.pt"
    if cache_path.exists():
        calibration = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        calibration = calibrate_specconlora(model, calibration_loader, loss_fn, gs)
        atomic_save(calibration, cache_path)
    with nonquantized_peft(model):
        model, report = prepare_specconlora_model(model, calibration_loader, loss_fn, gs, calibration=calibration)
    del calibration
    # PEFT serializes this switch: reloading the adapter must retain alpha/sqrt(r).
    if gs.scaling_mode != "sqrt_rank":
        raise ValueError("This configuration uses serializable RS-LoRA scaling")
    model.peft_config["default"].use_rslora = True
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" not in name:
                raise RuntimeError("Unexpected trainable base parameter: " + name)
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != gs.param_budget:
        raise RuntimeError(f"Actual SpecCon budget {trainable} != requested {gs.param_budget}")
    write_json(run / "speccon_report.json", report)
    write_json(run / "training_manifest.json", {
        "seed": seed, "base_params": base_params, "trainable_params": trainable,
        "total_unmerged_params": base_params + trainable, "speccon": asdict(gs),
        "train_examples": len(train_data), "eval_examples": len(eval_data),
        "train_fingerprint": train_data._fingerprint, "eval_fingerprint": eval_data._fingerprint,
        "calibration_examples": n_calibration, "dtype": "BF16 base / FP32 LoRA",
        "effective_batch_size": cfg["batch_size"] * cfg["gradient_accumulation_steps"],
        "backward": "author loop loss/accum, then Accelerate 1.4.0 loss/accum again",
        "checkpoint_for_mtbench": "final epoch (not best validation)",
        "device": torch.cuda.get_device_name(), "author_commit": manifest["author_commit"],
        "peft_compatibility": "temporarily skip unused bnb dispatch; ordinary PEFT Linear backend",
    })
    optimizer = optimizer_for(model, cfg)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    accumulation = cfg["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = updates_per_epoch * cfg["epochs"]
    scheduler = get_scheduler(cfg["scheduler"], optimizer=optimizer,
                              num_warmup_steps=int(total_steps * cfg["warmup_ratio"]),
                              num_training_steps=total_steps)
    state_path = run / "epoch_checkpoint.pt"
    epoch_start, completed, skipped = 0, 0, 0
    # Allocation uses its own stochastic operations; seed training explicitly.
    determinism.set_deterministic_environment(seed)
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        live = dict(accelerator.unwrap_model(model).named_parameters())
        for name, value in state["adapter"].items():
            live[name].data.copy_(value)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch_start, completed, skipped = state["epoch_next"], state["completed_steps"], state["skipped"]
        generator.set_state(state["data_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        del state
    print(json.dumps({"trainable": trainable, "total_steps": total_steps,
                      "resume_epoch": epoch_start, "seed": seed}), flush=True)
    optimizer.zero_grad()
    for epoch in range(epoch_start, cfg["epochs"]):
        model.train()
        loss_sum, finite_count, window_count = 0.0, 0, 0
        started = time.time()
        for index, batch in enumerate(train_loader):
            loss = model(**batch).loss
            if torch.isfinite(loss):
                loss_sum += float(loss.detach())
                finite_count += 1
                window_count += 1
                author_backward(accelerator, loss, accumulation)
            elif int((batch["labels"][..., 1:] != -100).sum()) == 0:
                skipped += 1
            else:
                raise RuntimeError("Non-finite training loss with supervised tokens")
            del loss
            if (index + 1) % accumulation == 0 or index == len(train_loader) - 1:
                if window_count:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    completed += 1
                    if completed % 100 == 0:
                        print(json.dumps({"epoch": epoch + 1, "step": completed, "total_steps": total_steps,
                                          "loss": loss_sum / finite_count, "lr": scheduler.get_last_lr(),
                                          "elapsed_epoch_seconds": time.time() - started}), flush=True)
                optimizer.zero_grad()
                window_count = 0
        metrics = evaluate(model, eval_loader, accelerator.device, cfg["eval_micro_batch_size"])
        metrics.update(epoch=epoch + 1, step=completed, train_loss=loss_sum / max(finite_count, 1),
                       skipped_empty_microbatches=skipped)
        write_json(run / f"epoch{epoch + 1}_metrics.json", metrics)
        print(json.dumps(metrics), flush=True)
        raw_model = accelerator.unwrap_model(model)
        atomic_save({"adapter": {n: p.detach().cpu() for n, p in raw_model.named_parameters() if p.requires_grad},
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "epoch_next": epoch + 1, "completed_steps": completed, "skipped": skipped,
                     "data_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
                     "numpy_rng": np.random.get_state()}, state_path)
    raw_model = accelerator.unwrap_model(model)
    raw_model.save_pretrained(run / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(run / "adapter")
    # Live scaling is retained through merge; no external tokenizer/runtime template.
    merged = raw_model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise RuntimeError("LoRA modules remain in merged checkpoint")
    merged.save_pretrained(run / "final_model", safe_serialization=True)
    tokenizer.save_pretrained(run / "final_model")
    write_json(run / "train_complete.json", {
        "seed": seed, "epochs": cfg["epochs"], "steps": completed, "planned_steps": total_steps,
        "trainable_params": trainable, "base_params": base_params,
        "final_weights_sha256": {p.name: sha256(p) for p in (run / "final_model").glob("*.safetensors")},
        "final_files_sha256": {p.name: sha256(p) for p in (run / "final_model").iterdir() if p.is_file()},
        "paid_judging": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    train(args.output.resolve(), args.seed)
