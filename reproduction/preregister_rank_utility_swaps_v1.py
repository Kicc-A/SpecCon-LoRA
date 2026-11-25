#!/usr/bin/env python3
"""Freeze MRPC rank swaps from A0 spectra, before any swap training metrics exist."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

PROJECT = Path('/workspace/SpecCon-LoRA')
OLD = PROJECT / 'experiments/extended_ablation_20261021_v2'
MECH = PROJECT / 'experiments/independent_calibration_mechanism_20261019_v1'
OUT = PROJECT / 'experiments/rank_utility_predictivity_v1'
RAW = MECH / 'runs/mrpc/a0_order0_raw.pt'
REFERENCE = OLD / 'runs/mrpc_near_uniform_svd_s42'
GAMMA = 1.2
LAMBDA = 1.5
BUDGET = 755712


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def leaf(name):
    return re.sub(r'^roberta\.encoder\.layer\.\d+\.', '', name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'pre_registered_swaps.json'
    if output.exists():
        raise RuntimeError('Pre-registration is frozen and already exists')
    raw = torch.load(RAW, map_location='cpu', weights_only=False)
    assert len(raw['ids']) == 512 and len(raw['blocks']) == 4
    original = json.loads((OLD / 'manifest.json').read_text())
    prior = next(j for j in original['jobs'] if j['spec']['task'] == 'mrpc'
        and j['spec']['seed'] == 42 and j['spec']['group'] == 'calibration_subset'
        and j['spec']['replicate'] == 0 and j['spec']['lambda'] == 1.5)
    assert raw['ids'] == prior['spec']['independent_ids']
    near = json.loads((REFERENCE / 'allocation.json').read_text())['rank_pattern']
    assert set(near) == set(raw['gradients']) == set(raw['raw_profiles'])
    original_calibration = json.loads((REFERENCE / 'calibration_rows.json').read_text())
    assert original_calibration['ids'] != raw['ids'], 'Historical near-uniform became reusable; re-audit'
    modules = {}
    for name in sorted(near):
        grad = raw['gradients'][name]['grad']
        profile = raw['raw_profiles'][name]
        mu = profile['mean_energy'].double().numpy()
        consistency = profile['consistency'].double().numpy()
        sv = torch.linalg.svdvals(grad.float())
        tail = float(sv[16:].square().sum())
        cost = int(sum(grad.shape))
        rank = int(near[name])
        assert 2 <= rank <= 16 and len(mu) == len(consistency) == 16
        score = {}
        for method, transformed in (
            ('raw_spectral_energy', mu),
            ('energy_only_utility', mu),
            ('consensus_adjusted_utility', mu * np.power(consistency, LAMBDA)),
        ):
            total = float(transformed.sum() + tail)
            if method == 'raw_spectral_energy':
                values = transformed / cost
            else:
                values = transformed * np.power(transformed / max(total, 1e-12), GAMMA-1) / cost
            score[method] = values.tolist()
        modules[name] = {'name': name, 'module_type': leaf(name),
            'shape': list(grad.shape), 'rank': rank, 'rank_cost': cost,
            'mean_gradient_tail_energy_after_16': tail,
            'score': score}
    assert sum(v['rank']*v['rank_cost'] for v in modules.values()) == BUDGET
    names = sorted(modules)
    candidates = []
    for receiver in names:
        a = modules[receiver]
        if a['rank'] >= 16:
            continue
        for donor in names:
            b = modules[donor]
            if donor == receiver or b['rank'] <= 2 or a['shape'] != b['shape']:
                continue
            new_a, new_b = a['rank']+1, b['rank']-1
            assert a['rank_cost'] == b['rank_cost']
            assert sum(v['rank']*v['rank_cost'] for v in modules.values()) + a['rank_cost']-b['rank_cost'] == BUDGET
            # Addition uses the next zero-based component; removal uses the
            # donor's highest currently allocated component, not receiver k.
            delta = {key: a['score'][key][a['rank']] - b['score'][key][b['rank']-1]
                     for key in a['score']}
            candidates.append({'receiver_module': receiver, 'donor_module': donor,
                'module_type': a['module_type'], 'shape': a['shape'],
                'receiver_rank_before': a['rank'], 'receiver_rank_after': new_a,
                'donor_rank_before': b['rank'], 'donor_rank_after': new_b,
                'addition_component_one_based': a['rank']+1,
                'removal_component_one_based': b['rank'],
                'rank_cost_each': a['rank_cost'], 'adapter_budget': BUDGET,
                'predicted_delta': delta})
    assert len(candidates) > 12
    for method in ('raw_spectral_energy', 'energy_only_utility', 'consensus_adjusted_utility'):
        order = sorted(range(len(candidates)), key=lambda i: candidates[i]['predicted_delta'][method], reverse=True)
        for rank, index in enumerate(order):
            candidates[index][method+'_rank'] = rank+1
    for item in candidates:
        item['energy_consensus_rank_gap'] = abs(item['energy_only_utility_rank'] -
                                                 item['consensus_adjusted_utility_rank'])
    with (OUT / 'all_feasible_swaps.csv').open('w', newline='') as stream:
        fields = ['receiver_module','donor_module','module_type','shape',
            'receiver_rank_before','donor_rank_before','rank_cost_each','adapter_budget',
            'raw_spectral_energy_delta','energy_only_utility_delta',
            'consensus_adjusted_utility_delta','energy_consensus_rank_gap']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow({**{key:item.get(key) for key in fields},
                **{method+'_delta':value for method,value in item['predicted_delta'].items()}})
    ordered = [
        ('consensus_high', sorted(candidates, key=lambda c:-c['predicted_delta']['consensus_adjusted_utility']),2),
        ('consensus_low', sorted(candidates, key=lambda c:c['predicted_delta']['consensus_adjusted_utility']),2),
        ('consensus_near_zero', sorted(candidates, key=lambda c:abs(c['predicted_delta']['consensus_adjusted_utility'])),2),
        ('energy_high', sorted(candidates, key=lambda c:-c['predicted_delta']['energy_only_utility']),2),
        ('largest_rank_disagreement', sorted(candidates, key=lambda c:-c['energy_consensus_rank_gap']),2),
        ('high_consensus_relative_to_energy', sorted(candidates,
            key=lambda c:c['energy_only_utility_rank']-c['consensus_adjusted_utility_rank'], reverse=True),2),
    ]
    selected = []
    used = {}
    for category, pool, count in ordered:
        added = 0
        for candidate in pool:
            if any(s['receiver_module']==candidate['receiver_module'] and
                   s['donor_module']==candidate['donor_module'] for s in selected):
                continue
            a,b = candidate['receiver_module'],candidate['donor_module']
            if used.get(a,0) >= 2 or used.get(b,0) >= 2:
                continue
            item = dict(candidate)
            item['selection_category'] = category
            item['swap_id'] = f'swap{len(selected)+1:02d}'
            selected.append(item)
            used[a] = used.get(a,0)+1
            used[b] = used.get(b,0)+1
            added += 1
            if added == count:
                break
        assert added == count, category
    assert len(selected) == 12
    allocations = {}
    for item in selected:
        ranks = dict(near)
        ranks[item['receiver_module']] = item['receiver_rank_after']
        ranks[item['donor_module']] = item['donor_rank_after']
        assert sum(ranks[n]*modules[n]['rank_cost'] for n in ranks) == BUDGET
        assert sum(ranks[n] != near[n] for n in ranks) == 2
        allocations[item['swap_id']] = ranks
    write(OUT / 'swap_rank_allocation.json', {'reference': near, 'swaps': allocations,
        'module_rank_costs': {n:m['rank_cost'] for n,m in modules.items()}})
    write(output, {'pre_registered_before_training': True, 'reference_calibration': 'A0 order0',
        'raw_gradients': str(RAW), 'raw_sha256': sha(RAW), 'old_reference': str(REFERENCE),
        'old_reference_reusable': False,
        'old_reference_mismatch': 'Historical near-uniform used nested prefix IDs; A0 cache changes SVD initialization',
        'model_initialization_seed': 42, 'training_seed': 42,
        'budget': BUDGET, 'gamma': GAMMA, 'lambda': LAMBDA,
        'scoring': 'Raw μ and allocator g(μ) and g(μ*C^λ), each divided by equal module rank cost; '
                   'S=sum first-16 adjusted block spectral energies + exact mean-gradient SVD tail beyond 16',
        'selection_rule': 'Six fixed categories x two swaps; maximum two selected uses per module',
        'all_feasible_count': len(candidates), 'selected': selected})
    print('PRE_REGISTERED', len(selected), 'of', len(candidates), 'valid swaps', flush=True)


if __name__ == '__main__':
    main()
