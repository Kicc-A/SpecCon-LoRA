"""Fail on incomplete adapter loads. Check FP32 layer algebra and BF16 full-model logits."""
import argparse
import gc
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source'))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.lora_modules import switch_to_lora, LinearWithGoRA
from tokenizer_assets import export_tokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--no-export', action='store_true')
    a = p.parse_args()
    run = a.run.resolve()
    assert (run / 'training_complete.json').exists(), 'Training did not finish successfully'
    cfg = Namespace(**json.loads((run / 'config.json').read_text()))
    ranks = json.loads((run / 'rank.json').read_text())
    raw = torch.load(run / 'final.ckpt', map_location='cpu', weights_only=False)
    sd = raw.get('model_state_dict', raw)
    expected = {f'{name}.{suffix}' for name in ranks for suffix in ('weight_a', 'weight_b')}
    assert len(ranks) == 128 and len(expected) == 256
    assert set(sd) == expected, {'missing': sorted(expected-set(sd)), 'unexpected': sorted(set(sd)-expected)}
    assert all(torch.isfinite(v).all() for v in sd.values())
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, torch_dtype=torch.bfloat16,
                                                 attn_implementation='sdpa', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, local_files_only=True)
    switch_to_lora(model, cfg)
    modules = {n:m for n,m in model.named_modules() if isinstance(m, LinearWithGoRA)}
    assert set(modules) == set(ranks)
    for name, m in modules.items():
        m.init_method = 'vanilla'
        m.dynamic_init(cfg.lora_rank, ranks[name])
        assert math.isclose(m.lora_scaler, 16 / math.sqrt(ranks[name]))
        assert tuple(sd[f'{name}.weight_a'].shape) == (ranks[name], m.in_features)
        assert tuple(sd[f'{name}.weight_b'].shape) == (m.out_features, ranks[name])
    result = model.load_state_dict(sd, strict=False)
    assert not result.unexpected_keys
    assert not (set(result.missing_keys) & expected)
    for name, m in modules.items():
        assert torch.equal(m.weight_a, sd[f'{name}.weight_a'])
        assert torch.equal(m.weight_b, sd[f'{name}.weight_b'])
    del raw, sd
    gc.collect()
    # Full inference on GPU 0; GPU 1 remains reserved by the parent lock.
    model.eval().to('cuda:0')
    prompts = ['Q: What is 13 plus 29?\nA: ', 'Q: A box holds 7 apples. How many apples do 3 boxes hold?\nA: ',
               'Q: There are 30 pencils. Five are given away. How many remain?\nA: ']
    inputs = [tokenizer(x, return_tensors='pt').to('cuda:0') for x in prompts]
    with torch.no_grad():
        before = [model(**x, use_cache=False).logits.float().cpu() for x in inputs]
        layer_checks = []
        generator = torch.Generator(device='cuda:0').manual_seed(12819)
        for name, m in modules.items():
            x = torch.randn((2,m.in_features), device='cuda:0', generator=generator)
            split = F.linear(F.linear(x, m.weight_a), m.weight_b) * m.lora_scaler
            dense = F.linear(x, (m.weight_b @ m.weight_a) * m.lora_scaler)
            err = (split-dense).norm() / split.norm().clamp_min(1e-15)
            assert err < 2e-4, (name, err.item())
            layer_checks.append(err.item())
            # One FP32 addition followed by BF16 rounding, matching FP32 author merge -> vLLM BF16.
            merged = (m.weight.float() + (m.weight_b @ m.weight_a) * m.lora_scaler).to(m.weight.dtype)
            m.weight.copy_(merged)
            m._del_lora()
        after = [model(**x, use_cache=False).logits.float().cpu() for x in inputs]
    comparisons = []
    for b, c in zip(before, after):
        rmse = (b-c).square().mean().sqrt().item()
        max_abs = (b-c).abs().max().item()
        kl = F.kl_div(c[:,-1].log_softmax(-1), b[:,-1].softmax(-1), reduction='batchmean').item()
        same = b[:,-1].argmax(-1).item() == c[:,-1].argmax(-1).item()
        # Fixed BF16 numerical gates. A failure must be investigated, not silently loosened.
        assert rmse < .08 and max_abs < 1.0 and kl < .005, (rmse,max_abs,kl)
        comparisons.append(dict(rmse=rmse, max_abs=max_abs, last_token_kl=kl, same_next_token=same))
    assert all(x['same_next_token'] for x in comparisons), comparisons
    report = dict(status='PASS', loaded_adapter_tensors=256, rank_modules=128,
                  adapter_values_exactly_equal_after_load=True,
                  fp32_layer_algebra_max_relative_error=max(layer_checks),
                  full_model_bf16_logit_checks=comparisons,
                  merge='FP32 W+sBA followed by BF16 cast; slight BF16 rounding is measured',
                  checkpoint=str(run / 'final.ckpt'))
    export = run / 'merged_hf'
    if not a.no_export:
        assert not export.exists(), f'Export exists: {export}; use --no-export to recheck'
        model.to('cpu')
        model.save_pretrained(export, safe_serialization=True, max_shard_size='5GB')
        report['tokenizer_export'] = export_tokenizer(cfg.model_name_or_path, export)
        # Save/reload verifies every exported parameter, independent of load_state_dict warnings.
        from safetensors import safe_open
        expected_sd = model.state_dict()
        seen = set()
        for shard in sorted(export.glob('*.safetensors')):
            with safe_open(shard, framework='pt', device='cpu') as f:
                for key in f.keys():
                    assert key in expected_sd and key not in seen
                    assert torch.equal(f.get_tensor(key), expected_sd[key]), key
                    seen.add(key)
        assert seen == set(expected_sd)
        report['export_parameter_tensors_verified'] = len(seen)
        report['export'] = str(export)
    (run / 'load_merge_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
