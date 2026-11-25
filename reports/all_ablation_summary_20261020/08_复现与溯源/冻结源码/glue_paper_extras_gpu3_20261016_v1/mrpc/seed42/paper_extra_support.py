"""Frozen preparation wrapper for R-Drop repeats and within-shape rank permutation."""
import copy
import importlib
import json
from pathlib import Path

from ablation_support import prepare_ablation_model


def prepare_paper_extra_model(model, loader, loss_fn, config, device, method,
                              reference, permutation_path, output):
    if permutation_path is None:
        return prepare_ablation_model(model, loader, loss_fn, config, device,
                                      method, reference, output)
    if method != 'speccon':
        raise ValueError('Rank permutation is defined for the full SpecCon configuration')
    meta = json.loads(Path(permutation_path).read_text())
    original = json.loads(Path(reference).read_text())
    assert original['rank_pattern'] == meta['original_rank_pattern']
    modules = dict(model.named_modules())
    for name, shape in meta['module_shapes'].items():
        assert list(modules[name].weight.shape) == shape, name
    calibration = importlib.import_module('speccon_lora.calibration')
    original_allocator = calibration.allocate_ranks
    called = []

    def permuted_allocator(grad_cache, allocation_config, **kwargs):
        ranks, stats = original_allocator(grad_cache, allocation_config, **kwargs)
        if ranks != meta['original_rank_pattern']:
            raise RuntimeError('Original allocation differs from the paired F reference')
        replacement = meta['permuted_rank_pattern']
        assert set(ranks) == set(replacement) == set(grad_cache)
        for name, item in grad_cache.items():
            assert list(item['grad'].shape) == meta['module_shapes'][name]
        used = sum(replacement[n] * sum(meta['module_shapes'][n]) for n in ranks)
        assert used == allocation_config.param_budget == 755712
        Path(output, 'unshuffled_allocation_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
        new_stats = {name: dict(module_name=name, rank=float(replacement[name]),
            original_rank=ranks[name], rank_cost=sum(meta['module_shapes'][name]),
            param_count=replacement[name]*sum(meta['module_shapes'][name]),
            allocation_source='within_shape_rank_permutation') for name in ranks}
        new_stats['__summary__'] = copy.deepcopy(stats['__summary__'])
        called.append(True)
        # The frozen calibration function now generates each module's SVD A from
        # its own gradient, using its new rank. No adapter tensors are moved.
        return dict(replacement), new_stats

    calibration.allocate_ranks = permuted_allocator
    try:
        prepared, report = prepare_ablation_model(model, loader, loss_fn, config,
            device, method, meta['permuted_allocation_path'], output)
    finally:
        calibration.allocate_ranks = original_allocator
    assert len(called) == 1 and len(report['applied_init']) == 72
    assert report['rank_pattern'] == meta['permuted_rank_pattern']
    assert all(item['scaling'] == 2.0 for item in report['applied_scaling'].values())
    path = Path(output, 'initial_state.json')
    initial = json.loads(path.read_text())
    initial.update(rank_permutation=str(permutation_path),
        permutation_seed=meta['permutation_seed'], changed_modules=meta['changed_modules'],
        original_allocation_verified=True, initialization='own_module_gradient_svd_at_permuted_rank')
    path.write_text(json.dumps(initial, indent=2)+'\n')
    Path(output, 'permutation.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('[RankPermutation]', json.dumps({'seed':meta['permutation_seed'],
        'changed_modules':meta['changed_modules'], 'adapter_parameters':755712}), flush=True)
    return prepared, report
