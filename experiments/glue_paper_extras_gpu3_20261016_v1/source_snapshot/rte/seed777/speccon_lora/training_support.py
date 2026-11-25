"""Shared calibration and held-out code validation for bounded LR experiments."""
import hashlib
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist


def distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if distributed() else 0


def barrier():
    if distributed():
        dist.barrier()


def rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])
    random.setstate(state["python"])


def cached_calibration(model, loader, loss_fn, config, args):
    from speccon_lora import calibrate_specconlora
    keys = ["model_name_or_path", "tokenizer_name_or_path", "train_dataset_name", "train_split",
            "train_question_column", "train_answer_column", "dataset_meta_prompt", "dataset_prefix",
            "dataset_postfix", "loraga_joint_tokenization", "no_pad_to_max_length", "dataset_preset",
            "max_length", "max_src_len", "max_train_samples", "seed", "bf16", "fp16"]
    signature = {"version": 1, "config": asdict(config), "args": {k: getattr(args, k, None) for k in keys},
                 "batch_size": loader.batch_size, "dataset_size": len(loader.dataset),
                 "world_size": dist.get_world_size() if distributed() else 1}
    data_path = Path(args.train_dataset_name)
    if data_path.is_file():
        stat = data_path.stat()
        signature["data_stat"] = [stat.st_size, stat.st_mtime_ns]
    signature["source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py") if p.name != "training_support.py"}
    # Normalize tuple/list differences and argparse numeric strings consistently.
    signature = json.loads(json.dumps(signature, sort_keys=True))
    path = Path(args.calibration_cache)
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["signature"] != signature:
            raise ValueError(f"Calibration cache signature mismatch: {path}")
        restore_rng(cached["rng_states"][rank()])
        if rank() == 0:
            print(f"[Calibration cache] Reused {path}", flush=True)
        return cached["calibration"]
    calibration = calibrate_specconlora(model, loader, loss_fn, config)
    state = rng_state()
    states = [None] * signature["world_size"]
    if distributed():
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"signature": signature, "rng_states": states, "calibration": calibration}, temporary)
        temporary.replace(path)
        print(f"[Calibration cache] Saved {path}", flush=True)
    barrier()
    return calibration


def code_validation_feature(example, feature, tokenizer, args):
    if not args.loraga_joint_tokenization:
        raise ValueError("Code validation masking requires joint tokenization")
    answer = str(example[args.train_answer_column])
    match = re.search(r"```(?:python|py)\s*\n(.*?)```", answer, flags=re.S | re.I)
    if match is None:
        raise ValueError("Validation response has no Python fenced code")
    prompt = args.dataset_meta_prompt + args.dataset_prefix + str(example[args.train_question_column]) + args.dataset_postfix
    full = prompt + answer + tokenizer.eos_token
    encoded = tokenizer(full, add_special_tokens=True, truncation=True, max_length=args.max_length,
                        return_offsets_mapping=True)
    if encoded["input_ids"] != feature["input_ids"]:
        raise ValueError("Validation tokenization differs from training")
    start, end = len(prompt) + match.start(1), len(prompt) + match.end(1)
    # Score only complete code tokens, retaining exactly the training input context.
    labels = list(feature["labels"])
    for i, (left, right) in enumerate(encoded["offset_mapping"]):
        if right <= left or left < start or right > end:
            labels[i] = -100
    return {**feature, "labels": labels}


class ValidationTracker:
    def __init__(self, loader, args):
        self.loader = loader
        self.output = Path(args.output_dir)
        self.interval = getattr(args, "validation_steps", 250)
        if loader is not None and self.interval <= 0:
            raise ValueError("validation_steps must be positive")
        self.last_step = None
        self.best = math.inf

    def maybe_evaluate(self, engine, step, final=False):
        if self.loader is None or step == self.last_step or (not final and step % self.interval):
            return
        state = rng_state()
        was_training = engine.module.training
        engine.module.eval()
        totals = torch.zeros(2, device=engine.device, dtype=torch.float64)
        try:
            with torch.inference_mode():
                for batch in self.loader:
                    batch = {k: v.to(engine.device) for k, v in batch.items()}
                    count = (batch["labels"][:, 1:] != -100).sum()
                    if count.item():
                        loss = engine.module(**batch).loss
                        totals[0] += loss.double() * count
                        totals[1] += count
            if distributed():
                dist.all_reduce(totals)
        finally:
            engine.module.train(was_training)
            restore_rng(state)
        nll = (totals[0] / totals[1]).item()
        if totals[1].item() <= 0 or not math.isfinite(nll):
            raise RuntimeError(f"Invalid validation totals: {totals.tolist()}")
        record = {"optimizer_step": step, "code_nll": nll, "scored_tokens": int(totals[1].item())}
        if rank() == 0:
            with (self.output / "validation_history.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[Validation] {json.dumps(record)}", flush=True)
        if nll < self.best:
            barrier()
            if rank() == 0:
                engine.module.save_pretrained(self.output / "best_adapter")
                (self.output / "best_validation.json").write_text(json.dumps(record, indent=2) + "\n")
            barrier()
            self.best = nll
        self.last_step = step
