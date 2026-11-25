"""Run the actual upstream trainer with assertions and evidence, not a replacement trainer."""
import hashlib
import json
import math
import os
import random
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import torch.distributed as dist
import common.lora_modules as package
import common.lora_modules.gora as g
from train.trainer import Trainer
from common.utils import to_device

PROFILE = os.environ['GORA_PROFILE']
OUT = Path(os.environ['GORA_RUN_DIR'])
if PROFILE == 'paper':
    from paper_changes import install
    install()

initialization = []
old_compress = g.LinearWithGoRA.grad_compress_init
def checked_compress(self, *args, **kwargs):
    gradient = self.weight.grad_stored.float().to(self.weight.device)
    assert torch.isfinite(gradient).all(), 'Non-finite stored gradient (check distributed buffer dtypes)'
    count = self.weight.iters if hasattr(self.weight, 'iters') else None
    result = old_compress(self, *args, **kwargs)
    A, B = self.weight_a.detach(), self.weight_b.detach()
    GAT = gradient @ A.T
    inv = torch.linalg.pinv(A @ A.T + 1e-8 * torch.eye(A.shape[0], device=A.device))
    gamma = kwargs['lr']
    expected = (GAT @ inv) * (gamma / math.sqrt(self.lora_rank/self.in_features)) * self.scale_rank / self.scaling_alpha
    if PROFILE == 'paper':
        expected.neg_()
    relative = ((B - expected).norm() / expected.norm().clamp_min(1e-20)).item()
    assert relative < 2e-4, f'Initialization formula mismatch: {relative}'
    inner_product = (B * GAT).sum().item() * self.lora_scaler
    assert math.isfinite(inner_product)
    assert inner_product <= 0 if PROFILE == 'paper' else inner_product >= 0
    initialization.append(dict(rank=self.lora_rank, scale=self.lora_scaler,
                               gradient_count=count, formula_relative_error=relative,
                               stored_gradient_dtype=str(self.weight.grad_stored.dtype),
                               gradient_update_inner_product=inner_product,
                               adapter_dtype=str(A.dtype)))
    return result
g.LinearWithGoRA.grad_compress_init = checked_compress

old_prepare = package.prepare_lora
def prepare(model, loader, args):
    assert args.world_size == 2
    assert args.batch_size_per_gpu * args.world_size * args.gradient_accumulation_steps == 64
    assert args.default_dtype == 'bf16' and args.lora_plus_scaler == 16
    assert args.dataset_seed == 42 and args.max_len == 1024
    modules = {n: m for n, m in model.named_modules() if isinstance(m, g.LinearWithGoRA)}
    assert len(modules) == 128
    assert all(n.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj')) for n in modules)
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    batch = to_device(next(iter(loader)), args.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        base_loss = float(model(**batch)[0])
    model.train(was_training)
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])
    old_prepare(model, loader, args)
    if args.global_rank == 0:
        importance = json.loads((OUT / 'importance.json').read_text())
        saved_ranks = json.loads((OUT / 'rank.json').read_text())
        scores = torch.tensor(list(importance.values()), dtype=torch.float32)
        proportions = (scores / scores.sum()).tolist()
        costs = {n: math.sqrt(m.in_features+m.out_features) if PROFILE == 'paper' else m.in_features+m.out_features
                 for n,m in modules.items()}
        budget = sum(costs.values()) * 8
        reference = {}
        for n, share in zip(importance, proportions):
            raw_rank = budget * share / costs[n] if PROFILE == 'paper' else round(budget * share) // costs[n]
            reference[n] = min(32,max(4,round(raw_rank)))
        assert reference == saved_ranks, 'Independent rank allocation check failed'
    counts = [x['gradient_count'] for x in initialization]
    assert len(initialization) == 128
    if args.global_rank == 0 and PROFILE == 'paper':
        assert set(counts) == {args.gradient_est_n_steps}, f'Gradient count mismatch {set(counts)}'
    model.eval()
    with torch.no_grad():
        initialized_loss = float(model(**batch)[0])
        saved_b = [m.weight_b.detach().clone() for m in modules.values()]
        for m in modules.values():
            m.weight_b.mul_(.001)
        tiny_loss = float(model(**batch)[0])
        for m, saved in zip(modules.values(), saved_b):
            m.weight_b.copy_(saved)
        del saved_b
    # Losses are diagnostics, no hyperparameter search; adapters restored bit for bit.
    model.train(was_training)
    assert all(math.isfinite(x) for x in (base_loss, initialized_loss, tiny_loss))
    digest = hashlib.sha256()
    total = 0
    for n, m in modules.items():
        assert 4 <= m.lora_rank <= 32
        assert abs(m.lora_scaler - 16 / math.sqrt(m.lora_rank)) < 1e-10
        for p in (m.weight_a, m.weight_b):
            assert p.dtype == torch.float32 and torch.isfinite(p).all()
            digest.update(p.detach().cpu().numpy().tobytes())
            total += p.numel()
    gathered = [None] * args.world_size
    dist.all_gather_object(gathered, digest.hexdigest())
    assert len(set(gathered)) == 1, f'Different initial adapters across workers: {gathered}'
    report = dict(status='PASS', profile=PROFILE, modules=128, adapter_tensors=256, adapter_parameters=total,
                  independent_rank_formula_check='128/128 modules match' if args.global_rank == 0 else 'checked on worker 0',
                  local_gradient_steps=args.gradient_est_n_steps,
                  gradient_count_range=[min(x for x in counts if x is not None), max(x for x in counts if x is not None)] if any(x is not None for x in counts) else None,
                  same_initial_adapter_hash_on_two_gpus=gathered,
                  probe_loss=dict(base=base_loss, initialized=initialized_loss, gamma_times_0_001=tiny_loss),
                  layers=initialization)
    (OUT / f'initialization.rank{args.global_rank}.json').write_text(json.dumps(report, indent=2) + '\n')
package.prepare_lora = prepare

old_train = Trainer.train
def checked_train(self, *args, **kwargs):
    engine = kwargs['model']
    groups = engine.optimizer.param_groups
    assert len(groups) == 2
    assert self.args.betas == [.9,.999] and self.args.weight_decay == .0005
    assert engine.train_batch_size() == 64
    scheduler = kwargs['lr_scheduler']
    assert sorted(scheduler.lr_scale) == [1,16], f'LoRA+ LR multipliers incorrect: {scheduler.lr_scale}'
    trainable = {n: p for n, p in engine.module.named_parameters() if p.requires_grad}
    assert len(trainable) == 256
    assert all(n.endswith(('weight_a', 'weight_b')) for n in trainable)
    initial = {n: p.detach().float().cpu().clone() for n, p in trainable.items()}
    old_forward = kwargs['forward_step']
    count = 0
    def forward(*fa, **fk):
        nonlocal count
        loss, metric = old_forward(*fa, **fk)
        assert torch.isfinite(loss).all(), 'Non-finite training loss'
        count += 1
        return loss, metric
    kwargs['forward_step'] = forward
    result = old_train(self, *args, **kwargs)
    assert count == self.args.num_micro_update_steps
    expected_steps = count // self.args.gradient_accumulation_steps
    assert engine.global_steps == expected_steps
    changes = {n: (p.detach().float().cpu() - initial[n]).norm().item() for n, p in trainable.items()}
    assert all(math.isfinite(v) for v in changes.values()) and all(v > 0 for v in changes.values())
    if self.args.global_rank == 0:
        report = dict(status='PASS', micro_steps=count, optimizer_steps=engine.global_steps,
                      pending_micro_steps=count % self.args.gradient_accumulation_steps,
                      changed_adapter_tensors=len(changes), parameter_change_norms=changes,
                      note='Upstream drops an incomplete final accumulation; preserved explicitly')
        (OUT / 'training_complete.json').write_text(json.dumps(report, indent=2) + '\n')
    dist.barrier()
    return result
Trainer.train = checked_train

if __name__ == '__main__':
    runpy.run_path(str(HERE / 'source/train/u_train.py'), run_name='__main__')
