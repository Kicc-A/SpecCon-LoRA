from pathlib import Path
import json
import torch
from safetensors import safe_open

root = Path(__file__).parent
base = Path('/root/.cache/modelscope/models/shakechen--Llama-2-7b-hf/snapshots/master')
index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
native = {}
mapping = {
    'self_attn.q_proj': 'attention.wq', 'self_attn.k_proj': 'attention.wk',
    'self_attn.v_proj': 'attention.wv', 'self_attn.o_proj': 'attention.wo',
    'mlp.gate_proj': 'feed_forward.w1', 'mlp.down_proj': 'feed_forward.w2',
    'mlp.up_proj': 'feed_forward.w3', 'input_layernorm': 'attention_norm',
    'post_attention_layernorm': 'ffn_norm',
}
for shard in sorted(set(index.values())):
    with safe_open(str(base / shard), framework='pt') as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            name = key
            if key == 'model.embed_tokens.weight':
                name = 'tok_embeddings.weight'
            elif key == 'model.norm.weight':
                name = 'norm.weight'
            elif key == 'lm_head.weight':
                name = 'output.weight'
            elif key.startswith('model.layers.'):
                name = key.removeprefix('model.')
                for old, new in mapping.items():
                    name = name.replace(old, new)
                if '.self_attn.q_proj.' in key or '.self_attn.k_proj.' in key:
                    original = value
                    value = value.reshape(32, 2, 64, 4096).transpose(1, 2).reshape(4096, 4096).contiguous()
                    reverse = value.reshape(32, 64, 2, 4096).transpose(1, 2).reshape(4096, 4096)
                    assert torch.equal(original, reverse)
            else:
                raise ValueError(key)
            native[name] = value
assert len(native) == len(index), (len(native), len(index))
torch.save(native, root / 'llama2_native.pth')
print('Converted 291 tensors, Q/K rotary layout round-trip verified.', flush=True)
