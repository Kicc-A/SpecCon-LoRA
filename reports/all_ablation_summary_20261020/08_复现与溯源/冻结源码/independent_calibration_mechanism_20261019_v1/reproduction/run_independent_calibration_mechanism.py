#!/usr/bin/env python3
"""Frozen, train-only mechanism experiment; no optimization or dev selection."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/workspace/SpecCon-LoRA/experiments/independent_calibration_mechanism_20261019_v1')
TASKS = {'cola': (320, 1.5, ['sentence']),
         'mrpc': (128, 1.5, ['sentence1', 'sentence2']),
         'rte': (256, 0.5, ['sentence1', 'sentence2'])}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def prepare(root):
    import numpy as np
    from datasets import load_from_disk
    if (root / 'manifest.json').exists():
        raise RuntimeError('Manifest already exists; use the frozen run command.')
    snapshot = root / 'source_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT / 'speccon_lora', snapshot / 'speccon_lora',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (snapshot / 'reproduction').mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot / 'reproduction' / Path(__file__).name)
    old = json.loads((PROJECT / 'reproduction/current_glue_manifest.json').read_text())
    manifest = dict(schema_version=1, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    model_path='/workspace/Models/RoBERTa-base', model_seed=42,
                    a_dropout_seed=7001, b_dropout_seed=8001, split_seed=191019,
                    samples=512, batch_size=32, blocks=4, replicates=5, orders=2,
                    budget=755712, tasks={}, source_hashes={})
    for task, (max_len, lam, fields) in TASKS.items():
        ds = load_from_disk('/workspace/datasets/glue/' + task)['train']
        fingerprints = [hashlib.sha256(json.dumps([str(row[k]).strip() for k in fields],
                                                 ensure_ascii=False).encode()).hexdigest()
                        for row in ds]
        rng = np.random.default_rng(manifest['split_seed'])
        heldout = rng.permutation(len(ds))[:512].tolist()
        heldout_keys = {fingerprints[i] for i in heldout}
        pool = [i for i in range(len(ds)) if fingerprints[i] not in heldout_keys]
        assert len(pool) >= 512
        replicas = []
        for rep in range(5):
            sample_seed = 192000 + rep
            chosen = np.random.default_rng(sample_seed).choice(pool, 512, replace=False).tolist()
            orders = []
            for order in range(2):
                order_seed = 193000 + 100 * rep + order
                ids = np.random.default_rng(order_seed).permutation(chosen).tolist()
                orders.append(dict(order=order, order_seed=order_seed, ids=ids,
                                   blocks=[ids[b:b+128] for b in range(0, 512, 128)]))
            replicas.append(dict(replicate=rep, sample_seed=sample_seed, orders=orders))
        ref = next(r for r in old['runs'] if r['task'] == task and r['seed'] == 42)
        matching = {}
        for f in (snapshot / 'speccon_lora').glob('*.py'):
            ref_file = Path(ref['cwd']) / 'speccon_lora' / f.name
            matching[f.name] = sha(f) == sha(ref_file)
        assert all(matching.values()), matching
        manifest['tasks'][task] = dict(train_rows=len(ds), max_length=max_len,
            consistency_lambda=lam, text_fields=fields, dataset_fingerprint=ds._fingerprint,
            heldout_ids=heldout, heldout_blocks=[heldout[b:b+128] for b in range(0, 512, 128)],
            heldout_label_counts={str(v): sum(int(ds[i]['label']) == v for i in heldout) for v in [0, 1]},
            available_a_rows=len(pool), duplicate_text_rows_excluded=len(ds)-512-len(pool),
            replicates=replicas, retained_reference=ref['source'],
            source_matches_retained=matching,
            a_pair_overlap_counts=[[len(set(a['orders'][0]['ids']) & set(b['orders'][0]['ids']))
                                    for b in replicas] for a in replicas])
    for f in sorted(snapshot.rglob('*.py')):
        manifest['source_hashes'][str(f.relative_to(root))] = sha(f)
    # Hash local model/tokenizer and dataset files; no remote artifacts involved.
    manifest['input_hashes'] = {}
    for directory in [Path(manifest['model_path'])] + [Path('/workspace/datasets/glue') / t for t in TASKS]:
        for f in sorted(directory.rglob('*')):
            if f.is_file() and (directory.name == 'RoBERTa-base' or '/train/' in str(f)) and not f.name.startswith('cache-'):
                manifest['input_hashes'][str(f)] = sha(f)
    dump(root / 'manifest.json', manifest)
    (root / 'PROTOCOL.md').write_text('''# 独立校准机制验证：预先冻结方案

CoLA、MRPC、RTE，模型/分类头种子42；仅用训练集，未读取dev/test样本。
每任务固定B=512条样本，仅用于机制评价；A为5次从剩余训练池无放回抽取的512条样本。
A重复之间允许重叠，完整交集计数在manifest；每个A均与B在行ID和完全相同文本层面隔离。
每个A做2种顺序/分块，共10次校准。每次16批×32、4块×128，与原校准相同。
模型权重不更新；普通CE，train模式保留基座dropout。A每次固定dropout种子7001，B为8001；
换序同时改变样本对应的dropout掩码，因此称“顺序/分块扰动”，不称纯分块因果实验。
谱随机种子3407、oversample8、power1；gamma1.2，rank[2,16]，严格参数预算755712。
lambda=0对照保留相同4块谱；完整方法lambda为CoLA/MRPC1.5、RTE0.5，沿用已有配置。
参考分配为attention rank4、FFN rank5（同预算）；另有100次同形状rank置换，仅为局部捕获参照。

主要读数（order0，5次校准）：
1. 评分预测性：C_A(k)与A/B块间平均左子空间重现度；同时报告C_B(k)。
   按相同k和矩阵形状分层，并残差化log平均块谱能量；再按module类型作更严格敏感性分析。
   报告每次校准的相关系数与描述性均值/标准差，不把72×16嵌套候选当独立实验。
2. 分配重复性：5个A两两的sum(cost*abs(rank差))/(2P)，越小越稳定；
   每组两种顺序的距离单列。两两距离相关，不将10对当10个独立样本做显著性检验。
3. 独立梯度捕获：对同一A的平均梯度做一次精确SVD，所有分配共用前16阶基底。
   左侧sum||U_A,r^T G_B||²/sum||G_B||²；右侧sum||G_B V_A,r||²/sum||G_B||²。
   右侧对应实际svd_a_zero_b的A行空间；两者都是初始点的局部梯度代理指标。
   另报捕获r_min=2之外的增量，避免基础rank掩盖差异。

保存样本ID、块划分、均值梯度、块基底/能量、原始与截断一致性、精确均值SVD基底/完整谱、
全部rank表、逐模块逐rank捕获/重现度、源码/输入哈希及运行环境。无优化器、无训练或调参。
先固定方案再查看结果，不保证一致性加权获胜；不得据此宣称下游准确率或跨模型泛化更好。
''')
    print(json.dumps({'prepared': str(root), 'tasks': list(TASKS), 'calibrations': 33}), flush=True)


def run(root, task):
    snapshot = root / 'source_snapshot'
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, set_seed
    import datasets
    from datasets import load_from_disk
    from speccon_lora.config import SpecConLoraConfig
    from speccon_lora.calibration import collect_consensus_gradients
    from speccon_lora.consensus import build_consensus_spectral_profiles
    from speccon_lora import rank_allocator as ra
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((root / 'manifest.json').read_text())
    for relative, expected in manifest['source_hashes'].items():
        assert sha(root / relative) == expected, relative
    spec = manifest['tasks'][task]
    work = root / 'runs' / task
    work.mkdir(parents=True, exist_ok=True)
    cfg = SpecConLoraConfig(target_modules=['query', 'key', 'value', 'dense'],
        target_prefix='roberta.encoder.layer', base_rank=8, r_min=2, r_max=16,
        lora_alpha=2, calibration_steps=16, calibration_blocks=4,
        use_subspace_consistency=True, subspace_consistency_lambda=spec['consistency_lambda'],
        subspace_consistency_floor=0.05, spectral_oversample=8, spectral_power_iterations=1,
        spectral_seed=3407, init_method='svd_a_zero_b', init_scale=1.0,
        scaling_mode='constant', rank_budget_mode='param', param_budget=755712,
        strict_param_budget=True, rank_score_gamma=1.2, attention_rank_prior=1.0)
    dump(work / 'config.json', dataclasses.asdict(cfg))
    dump(work / 'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        datasets=datasets.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(),
        started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), manifest_sha256=sha(root/'manifest.json')))
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
    raw = load_from_disk('/workspace/datasets/glue/' + task)['train']
    assert raw._fingerprint == spec['dataset_fingerprint']
    def tokenize(batch):
        out = tokenizer(*[batch[k] for k in spec['text_fields']], truncation=True,
                        max_length=spec['max_length'])
        out['labels'] = batch['label']
        return out
    # Read only train. Mapping whole train changes no model or RNG state.
    ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    set_seed(manifest['model_seed'])
    model = AutoModelForSequenceClassification.from_pretrained(manifest['model_path'], num_labels=2,
        problem_type='single_label_classification', local_files_only=True, dtype=torch.float32).cuda()
    def model_digest():
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    original_digest = model_digest()
    def collect(ids, dropout_seed, label):
        target = work / (label + '_raw.pt')
        if target.exists():
            print(f'[{task}/{label}] reuse raw', flush=True)
            return torch.load(target, map_location='cpu', weights_only=False)
        set_seed(dropout_seed)
        loader = DataLoader(ds.select(ids), batch_size=32, shuffle=False, collate_fn=collator,
                            generator=torch.Generator().manual_seed(9001))
        def loss_fn(current, batch):
            return current(**{k:v.cuda() for k,v in batch.items()}).loss
        start = time.time()
        names, grads, blocks, steps = collect_consensus_gradients(model, loader, loss_fn, cfg,
                                                                 average_distributed=False)
        assert len(names) == 72 and len(blocks) == 4 and steps == 16
        assert all(torch.isfinite(v['grad']).all() for v in grads.values())
        raw_profiles = build_consensus_spectral_profiles(blocks,
                         dataclasses.replace(cfg, subspace_consistency_floor=0.0))
        result = dict(ids=ids, dropout_seed=dropout_seed, gradients=grads,
            blocks=blocks, raw_profiles=raw_profiles, seconds=time.time()-start,
            label_counts={str(v):sum(int(raw[i]['label']) == v for i in ids) for v in [0,1]})
        torch.save(result, target.with_suffix('.tmp'))
        target.with_suffix('.tmp').replace(target)
        print(f'[{task}/{label}] calibrated {result["seconds"]:.1f}s', flush=True)
        return result
    b = collect(spec['heldout_ids'], manifest['b_dropout_seed'], 'heldout')
    b_total = sum(float(x['grad'].double().square().sum()) for x in b['gradients'].values())
    b_names = set(b['gradients'])
    validate_original = not (work / 'allocator_validation.json').exists()
    for replica in spec['replicates']:
        rep = replica['replicate']
        for order in replica['orders']:
            label = f'a{rep}_order{order["order"]}'
            if (work / (label + '_metrics.json')).exists():
                continue
            assert not set(order['ids']) & set(spec['heldout_ids'])
            a = collect(order['ids'], manifest['a_dropout_seed'], label)
            assert set(a['gradients']) == b_names
            start = time.time()
            spectra, bases, module_metrics = {}, {}, {}
            max_orth_error, max_energy_error = 0.0, 0.0
            for name, entry in a['gradients'].items():
                g = entry['grad'].cuda()
                # Preserve the original allocator's spectrum, including its numerical path.
                spectra[name] = ra._squared_svd_energy(g).cpu()
                # QR-based SVD gives orthonormal projection bases to tighter tolerance
                # than the default CUDA Jacobi solver on these ill-conditioned gradients.
                u, s, vh = torch.linalg.svd(g, full_matrices=False, driver='gesvd')
                u16, v16 = u[:, :16], vh[:16].T
                bases[name] = {'left': u16.cpu(), 'right': v16.cpu(), 'energy': spectra[name]}
                max_orth_error = max(max_orth_error, float((u16.T@u16-torch.eye(16,device='cuda')).abs().max()),
                                    float((v16.T@v16-torch.eye(16,device='cuda')).abs().max()))
                energy_error = abs(float(s.double().square().sum())-float(g.double().square().sum())) / max(float(g.double().square().sum()), 1e-30)
                max_energy_error = max(max_energy_error, energy_error)
                gb = b['gradients'][name]['grad'].cuda()
                left = (u16.T @ gb).double().square().sum(1).cumsum(0).cpu()
                right = (gb @ v16).double().square().sum(0).cumsum(0).cpu()
                gb_energy = float(gb.double().square().sum())
                overlaps = []
                for ba in a['blocks']:
                    for bb in b['blocks']:
                        overlap_matrix = (ba[name]['left'].T @ bb[name]['left']).square()
                        overlaps.append(overlap_matrix.cumsum(0).cumsum(1).diagonal() / torch.arange(1,17))
                cross = torch.stack(overlaps).mean(0)
                profile = a['raw_profiles'][name]
                module_metrics[name] = dict(shape=list(g.shape), cost=sum(g.shape),
                    mean_gradient_energy=float(g.double().square().sum()), b_gradient_energy=gb_energy,
                    mean_block_energy=profile['mean_energy'].tolist(), block_energy_std=profile['energy_std'].tolist(),
                    consistency_raw=profile['consistency'].tolist(),
                    consistency_used=profile['consistency'].clamp(0.05,1).tolist(),
                    b_consistency_raw=b['raw_profiles'][name]['consistency'].tolist(),
                    cross_subspace_recurrence=cross.tolist(), left_capture_energy=left.tolist(),
                    right_capture_energy=right.tolist())
                assert float(left[-1]) <= gb_energy * 1.0001
                assert float(right[-1]) <= gb_energy * 1.0001
                del g, gb, u, s, vh, u16, v16
            assert max_orth_error < 1e-4 and max_energy_error < 1e-4
            allocations = {}
            # Cache exact mean-gradient spectra only: call the original allocator unchanged.
            # This avoids recomputing full SVD for both lambdas; first run checks full path.
            original_svd_energy = ra._squared_svd_energy
            cached_by_ptr = {entry['grad'].data_ptr(): spectra[name] for name,entry in a['gradients'].items()}
            try:
                ra._squared_svd_energy = lambda g: cached_by_ptr[g.data_ptr()]
                for method, lam in [('energy', 0.0), ('consistency', spec['consistency_lambda'])]:
                    profiles = {n: dict(p, consistency=p['consistency'].clamp(0.05,1), consistency_lambda=lam)
                                for n,p in a['raw_profiles'].items()}
                    ranks, stats = ra.allocate_ranks(a['gradients'], dataclasses.replace(cfg, subspace_consistency_lambda=lam),
                                                    candidate_energy=profiles)
                    allocations[method] = dict(ranks=ranks, stats=stats)
            finally:
                ra._squared_svd_energy = original_svd_energy
            if validate_original:
                validation = {}
                for method, lam in [('energy', 0.0), ('consistency', spec['consistency_lambda'])]:
                    profiles = {n: dict(p, consistency=p['consistency'].clamp(0.05,1), consistency_lambda=lam)
                                for n,p in a['raw_profiles'].items()}
                    ranks, _ = ra.allocate_ranks(a['gradients'], dataclasses.replace(cfg, subspace_consistency_lambda=lam),
                                                candidate_energy=profiles)
                    validation[method] = ranks == allocations[method]['ranks']
                assert all(validation.values()), validation
                dump(work/'allocator_validation.json', validation)
                validate_original = False
            uniform = {n: 4 if m['shape'] == [768,768] else 5 for n,m in module_metrics.items()}
            allocations['near_uniform'] = dict(ranks=uniform)
            for method, alloc in allocations.items():
                ranks = alloc['ranks']
                cost = sum(module_metrics[n]['cost']*r for n,r in ranks.items())
                assert cost == 755712 and all(2 <= r <= 16 for r in ranks.values())
                alloc['budget_used'] = cost
                alloc['at_r_min_fraction'] = sum(r==2 for r in ranks.values())/72
                alloc['at_r_max_fraction'] = sum(r==16 for r in ranks.values())/72
                for side in ['left', 'right']:
                    captured = sum(module_metrics[n][side+'_capture_energy'][r-1] for n,r in ranks.items())
                    minimum = sum(m[side+'_capture_energy'][1] for m in module_metrics.values())
                    alloc[side+'_capture'] = captured/b_total
                    alloc[side+'_incremental_capture'] = (captured-minimum)/b_total
            shuffle_rng = np.random.default_rng(194000 + 100*rep + order['order'])
            groups = {}
            for name,m in module_metrics.items():
                groups.setdefault(tuple(m['shape']), []).append(name)
            shuffles = []
            for j in range(100):
                ranks = {}
                for names in groups.values():
                    values = shuffle_rng.permutation([allocations['consistency']['ranks'][n] for n in names])
                    ranks.update({n:int(r) for n,r in zip(names,values)})
                assert sum(module_metrics[n]['cost']*r for n,r in ranks.items()) == 755712
                shuffles.append(dict(index=j, ranks=ranks, **{side+'_capture':sum(
                    module_metrics[n][side+'_capture_energy'][r-1] for n,r in ranks.items())/b_total
                    for side in ['left','right']}))
            torch.save(bases, work / (label+'_mean_bases.pt'))
            result = dict(task=task, replicate=rep, order=order['order'], sample_seed=replica['sample_seed'],
                order_seed=order['order_seed'], model_sha256=original_digest, b_total_energy=b_total,
                label_counts=a['label_counts'], max_orthogonality_error=max_orth_error,
                max_spectral_energy_relative_error=max_energy_error, modules=module_metrics,
                allocations=allocations, shuffles=shuffles, calibration_seconds=a['seconds'],
                analysis_seconds=time.time()-start)
            dump(work/(label+'_metrics.json'), result)
            dump(work/'progress.json', dict(last_completed=label, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            print(json.dumps({'task':task,'done':label, 'left':{k:v['left_capture'] for k,v in allocations.items()},
                              'right':{k:v['right_capture'] for k,v in allocations.items()},
                              'analysis_seconds': result['analysis_seconds']}), flush=True)
            del a, bases
    assert model_digest() == original_digest, 'Model changed during calibration'
    artifacts = {p.name:sha(p) for p in sorted(work.glob('*')) if p.is_file() and p.name != 'DONE.json'}
    dump(work/'DONE.json', dict(task=task, complete=True, calibrations=11, a_runs=10,
        model_unchanged=True, model_sha256=original_digest, artifacts=artifacts,
        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(f'[{task}] COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','run'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--task', choices=list(TASKS))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root)
    else:
        if not args.task:
            parser.error('--task is required for run')
        run(args.root, args.task)
