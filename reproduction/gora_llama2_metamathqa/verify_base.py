"""Verify the HF checkpoint files against the pinned official repository."""
import hashlib
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
metadata = json.loads(Path(__file__).with_name('official_model.json').read_text())
assert metadata['id'] == 'meta-llama/Llama-2-7b-hf'
assets = {row['rfilename']: row for row in metadata['siblings']}
config = json.loads((base / 'config.json').read_text())
if config['vocab_size'] != 32000:
    sys.exit(f"Wrong base: vocab_size={config['vocab_size']}, expected 32000")
index = next((name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json') if (base / name).exists()), None)
if index is None:
    sys.exit('Missing checkpoint index')
suffix = '.bin' if index.startswith('pytorch') else '.safetensors'
shards = {name for name in assets if name.endswith(suffix) and '-of-' in name}
if set(json.loads((base / index).read_text())['weight_map'].values()) != shards:
    sys.exit('Unexpected weight map')
for name in sorted(shards | {'tokenizer.model'}):
    digest = hashlib.sha256()
    with (base / name).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != assets[name]['lfs']['sha256']:
        sys.exit(f'Checksum mismatch: {name}')
print('Official weight shards and tokenizer verified', metadata['sha'])
