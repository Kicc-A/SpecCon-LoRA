"""Preparation and runtime checks for the six-cell allocation/init ablation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from speccon_lora import prepare_specconlora_model, summarize_ranks


def variant_spec(variant, task_lambda):
    if variant not in 'ABCDEF' or len(variant) != 1:
        raise ValueError(f'Unknown variant: {variant}')
    uniform = variant in 'AB'
    return dict(
        ablation_variant=variant,
        allocation='uniform' if uniform else ('block_spectrum' if variant in 'CD' else 'consensus'),
        init_method='svd_a_zero_b' if variant in 'BDF' else 'none',
        adaptive_rank=not uniform,
        rank_range=[8, 8] if uniform else [2, 16],
        adapter_parameter_budget=1327104 if uniform else 755712,
        calibration_steps=0 if variant == 'A' else 16,
        calibration_blocks=0 if variant == 'A' else 4,
        gamma=None if uniform else 1.2,
        **{'lambda': None if uniform else (0.0 if variant in 'CD' else float(task_lambda))},
    )


def tensor_hash(named):
    checksum = hashlib.sha256()
    for name, parameter in sorted(named):
        checksum.update(name.encode())
        checksum.update(str(tuple(parameter.shape)).encode())
        checksum.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def prepare_main_ablation(model, loader, loss_fn, config, device, variant,
                          reference, expected_allocation, output):
    started = time.monotonic()
    spec = variant_spec(variant, config.subspace_consistency_lambda)
    original = json.loads(Path(reference).read_text())
    classifier_hash = tensor_hash([(n, p) for n, p in model.named_parameters() if 'classifier' in n])
    config.adaptive_rank = spec['adaptive_rank']
    config.init_method = spec['init_method']
    config.r_min, config.r_max = spec['rank_range']
    config.param_budget = spec['adapter_parameter_budget']
    # B also collects the same block-averaged gradients as the adaptive variants.
    # C/D keep consensus profiling enabled: lambda=0 changes only its weighting.
    config.use_subspace_consistency = variant != 'A'
    if variant in 'CD':
        config.subspace_consistency_lambda = 0.0
    if variant == 'A':
        config.calibration_steps = config.calibration_blocks = 0
    if variant in 'BCDEF' and (config.calibration_steps != 16 or config.calibration_blocks != 4):
        raise RuntimeError('Main ablation requires the same 16 calibration batches and 4 blocks')
    if config.scaling_mode != 'constant' or config.lora_alpha != 2:
        raise RuntimeError('Expected constant scaling 2')

    model, report = prepare_specconlora_model(model, loader, loss_fn, config, device=device)
    if set(report['target_names']) != set(original['target_names']):
        raise RuntimeError('Target modules changed from the paired baseline')
    if variant in 'AB':
        ranks = {name: 8 for name in report['target_names']}
        report['rank_pattern'] = ranks
        report['rank_summary'] = summarize_ranks(ranks)
        used = report['params']['trainable_params']
        report['rank_report'] = dict(budget_target=config.param_budget, budget_used=used,
            total_adapter_params=used, budget_unused=config.param_budget-used,
            rank_min=8, rank_max=8, rank_mean=8.0)
    if expected_allocation:
        expected = json.loads(Path(expected_allocation).read_text())
        if report['rank_pattern'] != expected['rank_pattern']:
            raise RuntimeError('Rank allocation differs from the matched reference')
    if variant in 'EF' and report['rank_pattern'] != original['rank_pattern']:
        raise RuntimeError('Consensus allocation differs from the original F cell')
    if variant in 'CD' and not all(
        values['used_consensus_spectrum'] and values['calibration_blocks'] == 4
        and values['subspace_consistency_lambda'] == 0
        for name, values in report['rank_stats'].items() if name != '__summary__'
    ):
        raise RuntimeError('Energy-only variant did not retain the block-spectrum path')
    expected_init_count = 72 if variant in 'BDF' else 0
    if len(report['applied_init']) != expected_init_count:
        raise RuntimeError('Unexpected spectral initialization count')
    if len(report['applied_scaling']) != 72 or any(
        values['scaling'] != 2.0 for values in report['applied_scaling'].values()
    ):
        raise RuntimeError('Unexpected module scaling')
    if variant in 'AB' and any(values['rank'] != 8 for values in report['applied_scaling'].values()):
        raise RuntimeError('Uniform allocation contains a non-rank8 adapter')

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    b_tensors = [(name, param) for name, param in trainable if 'lora_B' in name]
    if len(trainable) != 144 or any('lora_' not in name for name, _ in trainable):
        raise RuntimeError('Trainable modules differ from the frozen-head protocol')
    if sum(param.numel() for _, param in trainable) != config.param_budget:
        raise RuntimeError('Adapter parameter budget mismatch')
    if len(b_tensors) != 72 or any(torch.count_nonzero(param).item() for _, param in b_tensors):
        raise RuntimeError('All B matrices must initially be zero')
    if any(not torch.isfinite(param).all() for _, param in trainable):
        raise RuntimeError('Non-finite initialization')
    max_svd_gram_error = 0.0
    if variant in 'BDF':
        for name, param in trainable:
            if 'lora_A' in name:
                # Validate in CPU float64, independently of GPU matmul precision.
                basis = param.detach().to(device='cpu', dtype=torch.float64)
                gram = basis @ basis.T
                target = torch.eye(param.shape[0], dtype=torch.float64) * 0.5
                error = (gram - target).abs().max().item()
                max_svd_gram_error = max(max_svd_gram_error, error)
                # The frozen CUDA float32 SVD has small orthogonality residuals.
                # Keep its factors unchanged and record the measured residual.
                if not torch.allclose(gram, target, atol=1e-3, rtol=0.0):
                    raise RuntimeError(f'Spectral A is not the expected scaled orthonormal basis: {name}; max_error={error}; diagonal={gram.diag().tolist()}')
    output = Path(output)
    payload = dict(**spec, initial_adapter_sha256=tensor_hash(trainable),
        base_classifier_sha256=classifier_hash, all_B_zero=True,
        svd_initialized_modules=expected_init_count, target_modules=72,
        max_svd_gram_error=max_svd_gram_error,
        svd_gram_atol=1e-3,
        calibration_performed=variant != 'A',
        preparation_seconds=time.monotonic()-started,
        reference_rank_checked=bool(expected_allocation) or variant in 'EF')
    (output / 'initial_state.json').write_text(json.dumps(payload, indent=2) + '\n')
    (output / 'method_config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    print('[MainAblationInit]', json.dumps(payload), flush=True)
    return model, report
