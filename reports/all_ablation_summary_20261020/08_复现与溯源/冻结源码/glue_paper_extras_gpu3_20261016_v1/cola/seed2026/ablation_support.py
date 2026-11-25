"""Preparation-only changes; the frozen training loops remain unchanged."""
import hashlib
import json
import time
from pathlib import Path
import torch
from speccon_lora import prepare_specconlora_model, summarize_ranks


def tensor_hash(named):
    result = hashlib.sha256()
    for name, parameter in sorted(named):
        result.update(name.encode())
        result.update(str(tuple(parameter.shape)).encode())
        result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def prepare_ablation_model(model, loader, loss_fn, config, device, method, reference, output):
    start = time.monotonic()
    original = json.loads(Path(reference).read_text())
    classifier_before = tensor_hash([(n, p) for n, p in model.named_parameters() if "classifier" in n])
    if method == "uniform":
        config.adaptive_rank = False
        config.init_method = "none"  # PEFT's Kaiming A and zero B.
        config.r_min = config.r_max = config.base_rank
        config.use_subspace_consistency = False
    model, report = prepare_specconlora_model(model, loader, loss_fn, config, device=device)
    if set(report["target_names"]) != set(original["target_names"]):
        raise RuntimeError("Target modules differ from paired reference")
    if method == "uniform":
        ranks = {name: config.base_rank for name in report["target_names"]}
        report["rank_pattern"] = ranks
        report["rank_summary"] = summarize_ranks(ranks)
        used = report["params"]["trainable_params"]
        report["rank_report"] = {"budget_target": config.param_budget, "budget_used": used,
            "total_adapter_params": used, "budget_unused": config.param_budget - used,
            "rank_min": 8, "rank_max": 8, "rank_mean": 8.0}
        if report["applied_init"]:
            raise RuntimeError("Uniform baseline unexpectedly received spectral initialization")
        if any(value["rank"] != 8 for value in report["applied_scaling"].values()):
            raise RuntimeError("Uniform baseline contains a nonuniform module")
    elif report["rank_pattern"] != original["rank_pattern"]:
        raise RuntimeError("SpecCon rank pattern changed from paired R-Drop=1 reference")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    b_tensors = [(n, p) for n, p in trainable if "lora_B" in n]
    if len(b_tensors) != 72 or any(torch.count_nonzero(p).item() for _, p in b_tensors):
        raise RuntimeError("Expected exactly 72 zero-initialized B matrices")
    if len(trainable) != 144 or sum(p.numel() for _, p in trainable) != config.param_budget:
        raise RuntimeError("Adapter parameter count mismatch")
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError("Non-finite initial parameter")
    payload = {"adapter_method": method, "initial_adapter_sha256": tensor_hash(trainable),
        "base_classifier_sha256": classifier_before, "all_B_zero": True,
        "calibration_performed": method == "speccon", "target_modules": 72,
        "adapter_parameters": config.param_budget, "preparation_seconds": time.monotonic() - start,
        "gs_rank_matches_reference": method == "speccon"}
    Path(output, "initial_state.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("[AblationInit]", json.dumps(payload), flush=True)
    return model, report
