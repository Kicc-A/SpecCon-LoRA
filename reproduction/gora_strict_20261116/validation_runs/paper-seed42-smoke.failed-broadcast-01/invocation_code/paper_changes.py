"""Opt-in paper equations, never applied to the author profile.

Differences: negative projected gradient (3.3), round after dividing by sqrt cost
(3.2), parameter-specific post-accumulation gradient hooks, and adapter broadcast
(Appendix C.3, Algorithm 2). Keep other upstream training behavior.
"""
import math
import torch
import torch.distributed as dist

def install():
    import common.lora_modules.gora as g
    old_init = g.LinearWithGoRA.grad_compress_init
    def negative_init(self, *args, **kwargs):
        result = old_init(self, *args, **kwargs)
        with torch.no_grad():
            self.weight_b.neg_()
        return result
    g.LinearWithGoRA.grad_compress_init = negative_init

    # Replace only the two-stage rounding, preserving importance/scoring machinery.
    import inspect
    code = inspect.getsource(g.get_allocated_rank)
    old = ('smooth_trainable = allocate_func(smooth_total_budget * normalized_importance.item())\n'
           '                rank = smooth_trainable // named_smooth_features[name]')
    new = ('smooth_trainable = smooth_total_budget * normalized_importance.item()\n'
           '                rank = smooth_trainable / named_smooth_features[name]')
    assert code.count(old) == 1
    exec(compile(code.replace(old, new), '<paper-rank-equation>', 'exec'), g.__dict__)

    # Existing generic hook scans p.grad before the current parameter accumulates;
    # use a post-accumulation hook bound to each parameter to count exactly N.
    def factory(model, world_size=1, rank=0):
        def hook(p):
            grad = p.grad.detach()
            if world_size > 1:
                dist.all_reduce(grad)
                grad.div_(world_size)
            if rank == 0:
                value = grad.float().cpu()
                if not hasattr(p, 'grad_stored'):
                    p.grad_stored, p.iters = value.clone(), 1
                else:
                    p.grad_stored.add_(value)
                    p.iters += 1
            p.grad = None
        return hook
    g.get_record_gradient_hook = factory
    code = inspect.getsource(g.gora_reinit)
    old = 'module.weight.register_hook(get_record_gradient_hook(model, args.world_size, args.global_rank))'
    assert code.count(old) == 1
    exec(compile(code.replace(old, 'module.weight.register_post_accumulate_grad_hook(get_record_gradient_hook(model, args.world_size, args.global_rank))'),
                 '<paper-gradient-hooks>', 'exec'), g.__dict__)
    # prepare_lora imports the function into its package namespace.
    import common.lora_modules as package
    package.gora_reinit = g.gora_reinit

    old_dynamic = g.gora_dynamic_init
    def dynamic(*args, **kwargs):
        result = old_dynamic(*args, **kwargs)
        model = args[0] if args else kwargs['model']
        if dist.is_initialized():
            for module in model.modules():
                if isinstance(module, g.LinearWithGoRA):
                    dist.broadcast(module.weight_a.data, 0)
                    dist.broadcast(module.weight_b.data, 0)
        return result
    g.gora_dynamic_init = dynamic
