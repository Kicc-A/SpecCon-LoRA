try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        class _NoOpProgress:
            def update(self, *unused_args, **unused_kwargs):
                return None

            def close(self):
                return None

        return _NoOpProgress()

import math
import os
import time

import torch

from .rank_allocator import allocate_ranks
from .init import build_init_state
from .targets import find_target_module_names
from .consensus import build_block_spectral_profile, build_consensus_spectral_profiles


def clear_gradients(model):
    for param in model.parameters():
        param.grad = None


def _set_trainable_targets(model, target_set):
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if name in target_set:
            module.weight.requires_grad = True


def _average_distributed_gradient(grad_fp32):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(grad_fp32, op=torch.distributed.ReduceOp.SUM)
        grad_fp32.div_(torch.distributed.get_world_size())
    return grad_fp32


def capture_target_gradients(model, target_names, divisor=1, average_distributed=True):
    grad_cache = {}
    target_set = set(target_names)
    scale = float(max(int(divisor), 1))
    for name, module in model.named_modules():
        if name not in target_set:
            continue
        grad = getattr(module.weight, "grad", None)
        if grad is None:
            continue
        grad_fp32 = grad.detach().float().clone()
        if average_distributed:
            grad_fp32 = _average_distributed_gradient(grad_fp32)
        grad_cache[name] = {"grad": grad_fp32.div(scale).cpu()}
    return grad_cache


def _validate_consensus_config(config):
    if int(getattr(config, "calibration_blocks", 1)) < 2:
        raise ValueError("subspace consistency requires calibration_blocks >= 2")
    if float(getattr(config, "subspace_consistency_lambda", 1.0)) < 0:
        raise ValueError("subspace_consistency_lambda must be >= 0")
    floor = float(getattr(config, "subspace_consistency_floor", 0.05))
    if not 0.0 <= floor <= 1.0:
        raise ValueError("subspace_consistency_floor must be in [0, 1]")


def _consensus_budget_enabled(config):
    return (
        getattr(config, "rank_budget_mode", "independent") == "param"
        and int(getattr(config, "calibration_blocks", 1)) > 1
        and bool(getattr(config, "use_subspace_consistency", False))
    )


def collect_gradients(model, dataloader, loss_fn, config):
    target_names = find_target_module_names(
        model,
        config.target_modules,
        config.target_prefix,
    )
    target_set = set(target_names)

    _set_trainable_targets(model, target_set)
    model.train()
    clear_gradients(model)

    progress = tqdm(total=config.calibration_steps, desc="Calibrating gradients")
    for step, batch in enumerate(dataloader):
        if step >= config.calibration_steps:
            break
        loss = loss_fn(model, batch)
        (loss / config.calibration_steps).backward()
        progress.update(1)
    progress.close()
    grad_cache = capture_target_gradients(model, target_names, divisor=1)

    clear_gradients(model)
    for param in model.parameters():
        param.requires_grad = False

    if not grad_cache:
        raise ValueError("No gradients collected. Check target modules and loss_fn.")

    return target_names, grad_cache, [], None


def _add_grad_sums(total_sums, grad_sums):
    for name, item in grad_sums.items():
        grad = item["grad"]
        if name not in total_sums:
            total_sums[name] = {"grad": grad.clone()}
        else:
            total_sums[name]["grad"].add_(grad)


def _divide_grad_cache(grad_cache, divisor):
    scale = float(max(int(divisor), 1))
    return {
        name: {
            "grad": item["grad"].div(scale),
        }
        for name, item in grad_cache.items()
    }


def collect_consensus_gradients(
    model, dataloader, loss_fn, config, target_names_override=None,
    average_distributed=True,
):
    _validate_consensus_config(config)
    target_names = target_names_override
    if target_names is None:
        target_names = find_target_module_names(
            model,
            config.target_modules,
            config.target_prefix,
        )
    target_set = set(target_names)

    _set_trainable_targets(model, target_set)
    model.train()
    clear_gradients(model)

    calibration_steps = int(config.calibration_steps)
    calibration_blocks = int(getattr(config, "calibration_blocks", 1))
    block_size = max(1, math.ceil(max(calibration_steps, 1) / calibration_blocks))
    total_sums = {}
    block_profiles = []
    processed_steps = 0
    current_block_steps = 0

    def flush_block():
        nonlocal current_block_steps
        if current_block_steps <= 0:
            return
        block_sums = capture_target_gradients(
            model, target_names, divisor=1, average_distributed=average_distributed
        )
        _add_grad_sums(total_sums, block_sums)
        block_average = _divide_grad_cache(block_sums, current_block_steps)
        block_profiles.append(build_block_spectral_profile(block_average, config, len(block_profiles)))
        clear_gradients(model)
        current_block_steps = 0

    progress = tqdm(total=config.calibration_steps, desc="Calibrating gradients")
    for step, batch in enumerate(dataloader):
        if step >= calibration_steps:
            break
        if current_block_steps >= block_size and len(block_profiles) < calibration_blocks - 1:
            flush_block()
        loss = loss_fn(model, batch)
        loss.backward()
        processed_steps += 1
        current_block_steps += 1
        progress.update(1)
    progress.close()
    flush_block()

    grad_cache = _divide_grad_cache(total_sums, processed_steps) if processed_steps > 0 else {}

    clear_gradients(model)
    for param in model.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not grad_cache:
        raise ValueError("No gradients collected. Check target modules and loss_fn.")

    return target_names, grad_cache, block_profiles, processed_steps



def _finalize_calibration(target_names, grad_cache, block_profiles, processed_steps, config):
    candidate_profiles = None
    if block_profiles:
        print(
            f"Collected gradients for {len(grad_cache)} LoRA target matrices "
            f"from {processed_steps} calibration batches",
            flush=True,
        )
        start = time.time()
        print(
            f"Building cross-block subspace consensus scores from {len(block_profiles)} calibration blocks...",
            flush=True,
        )
        candidate_profiles = build_consensus_spectral_profiles(block_profiles, config)
        block_profiles = []
        print(f"Consensus spectral scores built in {time.time() - start:.1f}s", flush=True)

    if config.adaptive_rank:
        start = time.time()
        print("Allocating adaptive ranks...", flush=True)
        rank_pattern, rank_stats = allocate_ranks(grad_cache, config, candidate_energy=candidate_profiles)
        print(f"Rank allocation finished in {time.time() - start:.1f}s", flush=True)
        summary = rank_stats.get("__summary__") if isinstance(rank_stats, dict) else None
        if summary:
            print(f"Rank report summary: {summary}", flush=True)
    else:
        rank_pattern = {name: int(config.base_rank) for name in target_names if name in grad_cache}
        rank_stats = {}

    start = time.time()
    print(f"Building init state with method={config.init_method}...", flush=True)
    init_state = build_init_state(grad_cache, rank_pattern, config)
    print(f"Init state built in {time.time() - start:.1f}s", flush=True)
    return {
        "target_names": target_names,
        "rank_pattern": rank_pattern,
        "rank_stats": rank_stats,
        "init_state": init_state,
    }


def calibrate_specconlora(model, dataloader, loss_fn, config):
    sharded = (
        os.environ.get("SPECCON_CALIBRATION_SHARDED", "0") == "1"
        and _consensus_budget_enabled(config)
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )
    if sharded:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        target_names = find_target_module_names(model, config.target_modules, config.target_prefix)
        local_target_names = target_names[rank::world_size]
        _, local_grad_cache, local_block_profiles, processed_steps = collect_consensus_gradients(
            model,
            dataloader,
            loss_fn,
            config,
            target_names_override=local_target_names,
            average_distributed=False,
        )
        cache_prefix = os.environ.get("SPECCON_CALIBRATION_CACHE")
        if not cache_prefix:
            raise ValueError("SPECCON_CALIBRATION_CACHE is required for sharded calibration")
        partial_path = f"{cache_prefix}.rank{rank}.pt"
        final_path = f"{cache_prefix}.final.pt"
        torch.save(
            {
                "grad_cache": local_grad_cache,
                "block_profiles": local_block_profiles,
                "processed_steps": processed_steps,
            },
            partial_path,
        )
        del local_grad_cache, local_block_profiles
        torch.distributed.barrier()
        if rank == 0:
            grad_cache = {}
            block_profiles = []
            for shard_rank in range(world_size):
                payload = torch.load(
                    f"{cache_prefix}.rank{shard_rank}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                grad_cache.update(payload["grad_cache"])
                while len(block_profiles) < len(payload["block_profiles"]):
                    block_profiles.append({})
                for index, profile in enumerate(payload["block_profiles"]):
                    block_profiles[index].update(profile)
                processed_steps = payload["processed_steps"]
                del payload
            calibration = _finalize_calibration(
                target_names, grad_cache, block_profiles, processed_steps, config
            )
            torch.save(calibration, final_path)
        torch.distributed.barrier()
        if rank != 0:
            calibration = torch.load(final_path, map_location="cpu", weights_only=False)
        torch.distributed.barrier()
        if rank == 0:
            for shard_rank in range(world_size):
                os.remove(f"{cache_prefix}.rank{shard_rank}.pt")
            os.remove(final_path)
        return calibration

    if _consensus_budget_enabled(config):
        target_names, grad_cache, block_profiles, processed_steps = collect_consensus_gradients(
            model, dataloader, loss_fn, config
        )
    else:
        target_names, grad_cache, _, _ = collect_gradients(model, dataloader, loss_fn, config)
        print(f"Collected gradients for {len(grad_cache)} LoRA target matrices", flush=True)
        block_profiles = []
        processed_steps = int(config.calibration_steps)
    return _finalize_calibration(
        target_names, grad_cache, block_profiles, processed_steps, config
    )
