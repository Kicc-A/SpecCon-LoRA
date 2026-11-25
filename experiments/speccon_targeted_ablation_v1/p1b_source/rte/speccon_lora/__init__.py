from .config import SpecConLoraConfig
from .report import count_trainable_params, summarize_ranks
from .targets import find_target_module_names, is_target_module


def calibrate_specconlora(model, dataloader, loss_fn, config: SpecConLoraConfig):
    from .calibration import calibrate_specconlora as _calibrate_specconlora

    return _calibrate_specconlora(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        config=config,
    )


def inject_specconlora(model, config, target_names, rank_pattern=None, init_state=None):
    from .peft_injector import inject_specconlora as _inject_specconlora

    return _inject_specconlora(
        model=model,
        config=config,
        target_names=target_names,
        rank_pattern=rank_pattern,
        init_state=init_state,
    )


def prepare_specconlora_model(model, dataloader, loss_fn, config: SpecConLoraConfig, device=None, calibration=None):
    needs_calibration = config.adaptive_rank or config.init_method != "none"
    if needs_calibration:
        if calibration is None:
            calibration = calibrate_specconlora(
                model=model, dataloader=dataloader, loss_fn=loss_fn, config=config,
            )
        target_names = calibration["target_names"]
        rank_pattern = calibration["rank_pattern"] if config.adaptive_rank else {}
        rank_stats = calibration["rank_stats"]
        init_state = calibration["init_state"]
    else:
        target_names = find_target_module_names(
            model,
            config.target_modules,
            config.target_prefix,
        )
        rank_pattern = {}
        rank_stats = {}
        init_state = {}

    model, applied_init, applied_scaling = inject_specconlora(
        model=model,
        config=config,
        target_names=target_names,
        rank_pattern=rank_pattern,
        init_state=init_state,
    )
    report = {
        "target_names": target_names,
        "rank_pattern": rank_pattern,
        "rank_stats": rank_stats,
        "rank_report": rank_stats.get("__summary__") if isinstance(rank_stats, dict) else None,
        "rank_summary": summarize_ranks(rank_pattern),
        "init_modules": sorted(init_state),
        "svd_init_modules": sorted(init_state) if config.init_method.startswith("svd_") else [],
        "applied_init": applied_init,
        "applied_scaling": applied_scaling,
        "params": count_trainable_params(model),
    }
    return model, report


__all__ = [
    "SpecConLoraConfig",
    "calibrate_specconlora",
    "count_trainable_params",
    "find_target_module_names",
    "inject_specconlora",
    "is_target_module",
    "prepare_specconlora_model",
    "summarize_ranks",
]
