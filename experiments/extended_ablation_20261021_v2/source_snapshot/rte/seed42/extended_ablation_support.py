"""Controlled allocation, calibration and initialization extensions to frozen GLUE runners."""
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch


def read_spec(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def tensor_hash(named):
    h = hashlib.sha256()
    for name, value in sorted(named):
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare_extended(model, loader, loss_fn, config, device, spec_path, reference, output):
    from speccon_lora import prepare_specconlora_model, summarize_ranks

    started = time.monotonic()
    spec = read_spec(spec_path)
    ref = read_spec(reference)
    config.param_budget = spec["budget"]
    config.rank_score_gamma = spec["gamma"]
    config.calibration_steps = spec["steps"]
    config.calibration_blocks = spec["blocks"]
    config.subspace_consistency_lambda = spec["lambda"]
    config.init_method = "none" if spec["init"] == "random" else "svd_a_zero_b"
    config.adaptive_rank = True
    config.use_subspace_consistency = True
    assert config.scaling_mode == "constant" and config.lora_alpha == 2
    plain_uniform = spec["allocation"] == "uniform" and spec["init"] == "random"
    if plain_uniform:
        assert spec["steps"] == spec["blocks"] == 0
        config.adaptive_rank = False
        config.base_rank = config.r_min = config.r_max = spec["uniform_rank"]
        config.use_subspace_consistency = False
    else:
        assert spec["blocks"] >= 2 and spec["steps"] % spec["blocks"] == 0
    classifier_before = tensor_hash([(n, p) for n, p in model.named_parameters() if "classifier" in n])
    calibration = importlib.import_module("speccon_lora.calibration")
    injector = importlib.import_module("speccon_lora.peft_injector")
    allocate_original = calibration.allocate_ranks
    collect_original = calibration.collect_consensus_gradients
    init_original = injector.apply_svd_init
    allocation_details = {}
    init_details = {}

    def allocate(gradients, cfg, **kwargs):
        if spec["allocation"] == "adaptive":
            ranks, stats = allocate_original(gradients, cfg, **kwargs)
        else:
            ranks = {}
            for name in gradients:
                if spec["allocation"] == "near_uniform":
                    ranks[name] = 4 if ".attention." in name else 5
                elif spec["allocation"] == "uniform":
                    ranks[name] = spec["uniform_rank"]
                else:
                    raise ValueError(spec["allocation"])
            used = sum(ranks[n] * sum(v["grad"].shape) for n, v in gradients.items())
            assert used == cfg.param_budget, (used, cfg.param_budget)
            stats = {n: {"rank": r, "allocation": spec["allocation"]} for n, r in ranks.items()}
            stats["__summary__"] = dict(budget_target=used, budget_used=used,
                total_adapter_params=used, budget_unused=0,
                rank_min=min(ranks.values()), rank_max=max(ranks.values()),
                rank_mean=sum(ranks.values()) / len(ranks))
        allocation_details.update(stats)
        if spec.get("expect_reference_allocation"):
            assert ranks == ref["rank_pattern"], "Changed allocation in a fixed-allocation control"
        return ranks, stats

    def collect(current, data, fn, cfg, **kwargs):
        if spec.get("independent_raw"):
            payload = torch.load(spec["independent_raw"], map_location="cpu", weights_only=False)
            assert payload["ids"] == spec["independent_ids"]
            assert payload["dropout_seed"] == 7001
            assert len(payload["blocks"]) == cfg.calibration_blocks == 4
            assert len(payload["ids"]) == 32 * cfg.calibration_steps == 512
            assert set(payload["gradients"]) == set(ref["target_names"])
            modules = dict(current.named_modules())
            assert all(tuple(modules[n].weight.shape) == tuple(v["grad"].shape)
                       for n, v in payload["gradients"].items())
            # The cached gradients are from the same seed42 model and frozen source.
            # Initialization random numbers are explicitly fixed for both lambda arms.
            torch.manual_seed(7001)
            torch.cuda.manual_seed_all(7001)
            for p in current.parameters():
                p.requires_grad = False
            return list(ref["target_names"]), payload["gradients"], payload["blocks"], 16
        result = collect_original(current, data, fn, cfg, **kwargs)
        assert result[3] == spec["steps"]
        assert len(result[2]) == spec["blocks"]
        return result

    def initialize(current, state, method="svd_a_zero_b"):
        for name, module in current.named_modules():
            if not hasattr(module, "lora_A") or "default" not in module.lora_A:
                continue
            original_name = next((n for n in state if name.endswith(n)), None)
            if original_name is None:
                continue
            random_a = module.lora_A["default"].weight.detach().cpu().clone()
            svd_a = state[original_name]["lora_A"].clone()
            chosen = svd_a
            if spec["init"] == "random_svdnorm":
                chosen = random_a * (svd_a.norm() / random_a.norm())
            elif spec["init"] == "svd_randomnorm":
                chosen = svd_a * (random_a.norm() / svd_a.norm())
            elif spec["init"] == "orthogonal_svdnorm":
                seed = int(hashlib.sha256((str(spec["seed"]) + original_name).encode()).hexdigest()[:12], 16)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                q, _ = torch.linalg.qr(torch.randn(svd_a.shape[1], svd_a.shape[0],
                    dtype=torch.float64, generator=generator), mode="reduced")
                chosen = q.T.float() * (svd_a.norm() / (svd_a.shape[0] ** 0.5))
            state[original_name]["lora_A"] = chosen
            gram = chosen.double() @ chosen.double().T
            expected_norm = random_a.norm() if spec["init"] == "svd_randomnorm" else svd_a.norm()
            assert torch.allclose(chosen.norm(), expected_norm, atol=2e-6, rtol=2e-6)
            gram_error = float((gram - torch.eye(len(chosen), dtype=torch.float64) *
                float(chosen.square().sum() / len(chosen))).abs().max())
            if spec["init"] == "orthogonal_svdnorm":
                assert gram_error < 1e-5
            init_details[original_name] = dict(random_frobenius=float(random_a.norm()),
                svd_frobenius=float(svd_a.norm()), actual_frobenius=float(chosen.norm()),
                isotropic_gram_error=gram_error)
        return init_original(current, state, method=method)

    calibration.allocate_ranks = allocate
    calibration.collect_consensus_gradients = collect
    injector.apply_svd_init = initialize
    try:
        model, report = prepare_specconlora_model(model, loader, loss_fn, config, device=device)
    finally:
        calibration.allocate_ranks = allocate_original
        calibration.collect_consensus_gradients = collect_original
        injector.apply_svd_init = init_original
    if plain_uniform:
        ranks = {name: spec["uniform_rank"] for name in report["target_names"]}
        report["rank_pattern"] = ranks
        report["rank_summary"] = summarize_ranks(ranks)
        used = report["params"]["trainable_params"]
        report["rank_report"] = dict(budget_target=spec["budget"], budget_used=used,
            total_adapter_params=used, budget_unused=spec["budget"]-used,
            rank_min=spec["uniform_rank"], rank_max=spec["uniform_rank"],
            rank_mean=float(spec["uniform_rank"]))
    assert set(report["target_names"]) == set(ref["target_names"])
    assert len(report["rank_pattern"]) == 72
    assert report["rank_report"]["budget_used"] == spec["budget"]
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 144
    assert all("lora_" in n for n, _ in trainable)
    assert sum(p.numel() for _, p in trainable) == spec["budget"]
    assert all(torch.isfinite(p).all() for _, p in trainable)
    assert all(torch.count_nonzero(p) == 0 for n, p in trainable if "lora_B" in n)
    assert all(x["scaling"] == 2.0 for x in report["applied_scaling"].values())
    classifier_after = tensor_hash([(n.replace("base_model.model.", "").replace(
        ".original_module", "").replace(".modules_to_save.default", ""), p)
        for n, p in model.named_parameters()
        if "classifier" in n and ".modules_to_save." not in n])
    assert classifier_before == classifier_after, "Classifier changed during preparation"
    if spec.get("expected_initial_adapter_sha256"):
        assert tensor_hash(trainable) == spec["expected_initial_adapter_sha256"], "Initial weights differ from paired baseline"
    write_json(Path(output) / "initial_state.json", dict(spec=spec, config=asdict(config),
        classifier_sha256=classifier_before, adapter_sha256=tensor_hash(trainable),
        zero_B_verified=True, initialization=init_details, setup_seconds=time.monotonic()-started))
    write_json(Path(output) / "allocation_details.json", allocation_details)
    torch.save({n: p.detach().cpu() for n, p in trainable}, Path(output) / "initial_adapter.pt")
    # Record exact rows even in smoke runs; the regular calibrations use nested prefixes.
    write_json(Path(output) / "calibration_rows.json", dict(
        ids=spec.get("independent_ids", list(range(spec["steps"] * 32))),
        blocks=spec["blocks"], steps=spec["steps"], dataset_split="train",
        source=spec.get("independent_raw", "nested_train_prefix")))
    return model, report
