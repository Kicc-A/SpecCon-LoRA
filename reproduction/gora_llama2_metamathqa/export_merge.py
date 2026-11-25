"""Export using the merge procedure in GoRA scripts/run_vllm.py."""
import argparse
import json
import shutil
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from common.lora_modules import LinearWithGoRA, LinearWithLoRA, switch_to_lora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    assert (run / 'final.ckpt').exists()
    config = Namespace(**json.loads((run / 'config.json').read_text()))
    ranks = json.loads((run / 'rank.json').read_text())
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, local_files_only=True, torch_dtype=torch.float32)
    switch_to_lora(model, config)
    modules = {name: module for name, module in model.named_modules() if isinstance(module, LinearWithGoRA)}
    assert set(modules) == set(ranks) and len(modules) == 224
    for name, module in modules.items():
        module.init_method = 'vanilla'
        module.dynamic_init(config.lora_rank, ranks[name])
    raw = torch.load(run / 'final.ckpt', map_location='cpu', weights_only=False)
    state = raw.get('model_state_dict', raw)
    expected = {f'{name}.{suffix}' for name in ranks for suffix in ('weight_a', 'weight_b')}
    assert set(state) == expected and all(torch.isfinite(value).all() for value in state.values())
    loaded = model.load_state_dict(state, strict=False)
    assert not loaded.unexpected_keys and not (set(loaded.missing_keys) & expected)
    assert all(torch.equal(module.weight_a, state[f'{name}.weight_a']) and torch.equal(module.weight_b, state[f'{name}.weight_b']) for name, module in modules.items())
    for _, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            module.merge_and_del()
    export = run / 'merged_hf'
    export.mkdir(exist_ok=False)
    model.save_pretrained(export, safe_serialization=True, max_shard_size='5GB')
    for name in ('tokenizer.model', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json'):
        source = Path(config.model_name_or_path) / name
        if source.exists():
            shutil.copyfile(source, export / name)
    report = {'source_checkpoint': str(run / 'final.ckpt'), 'merged_model': str(export), 'rank_modules': len(modules), 'adapter_tensors': len(state), 'merge_method': 'GoRA scripts/run_vllm.py get_sd_from_mt_lora: switch_to_lora, dynamic_init, load adapter, merge_and_del'}
    (run / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
