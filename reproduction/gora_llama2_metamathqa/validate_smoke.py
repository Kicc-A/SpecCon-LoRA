"""Check that the short run exercised the paper settings and saved an adapter."""
import json
import math
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
run = root / 'smoke-seed42'
config = json.loads((run / 'config.json').read_text())
ranks = json.loads((run / 'rank.json').read_text())
importance = json.loads((run / 'importance.json').read_text())
log = (root / 'smoke.log').read_text()
assert config['gradient_est_n_steps'] == 64
assert config['gora_lr'] == 0.08
assert config['gora_features_func'] == 'sqrt'
assert len(ranks) == len(importance) == 224
assert 'GoRA gradient computing step: 64,' in log
assert 'Traceback (most recent call last)' not in log

model_config = json.loads((Path(config['model_name_or_path']) / 'config.json').read_text())
hidden = model_config['hidden_size']
kv_hidden = hidden * model_config['num_key_value_heads'] // model_config['num_attention_heads']
def features(name):
    if name.endswith(('gate_proj', 'up_proj', 'down_proj')):
        return hidden + model_config['intermediate_size']
    return hidden + (kv_hidden if name.endswith(('k_proj', 'v_proj')) else hidden)

total_budget = sum(math.sqrt(features(name)) * config['lora_rank'] for name in importance)
normalized = torch.tensor(list(importance.values()))
normalized = normalized / normalized.sum()
expected = {}
for (name, _), share in zip(importance.items(), normalized):
    rank = round(total_budget * share.item() / math.sqrt(features(name)))
    expected[name] = min(config['gora_max_rank'], max(config['gora_min_rank'], rank))
assert ranks == expected, {'different': sum(ranks[n] != expected[n] for n in ranks)}

checkpoint = torch.load(run / 'final.ckpt', map_location='cpu', weights_only=False)
state = checkpoint.get('model_state_dict', checkpoint)
keys = {f'{name}.{suffix}' for name in ranks for suffix in ('weight_a', 'weight_b')}
assert set(state) == keys and all(torch.isfinite(value).all() for value in state.values())
report = {'status': 'PASS', 'gradient_steps': 64, 'rank_modules': len(ranks),
          'rank_map_matches_paper_formula': True, 'adapter_tensors': len(state),
          'parameters': sum(ranks[name] * features(name) for name in ranks),
          'smoke_optimizer_steps': 2}
(run / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
