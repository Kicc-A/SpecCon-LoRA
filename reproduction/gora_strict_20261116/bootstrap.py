"""Freeze the verified upstream revision; never modify the user's working repositories."""
import difflib
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = '4037d4d6ba67ff88de87f90b943ff4e3a3649b67'
UPSTREAM = Path('/workspace/MyTransformers')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    source = HERE / 'source'
    if source.exists():
        assert (HERE / 'source_manifest.json').exists(), 'Incomplete bootstrap; inspect source first'
        print('Source already frozen. Use preflight.py to verify it.')
        return
    archive = HERE / 'upstream.tar'
    subprocess.run(['git', '-C', str(UPSTREAM), 'archive', '--format=tar', COMMIT, '-o', str(archive)], check=True)
    source.mkdir()
    with tarfile.open(archive) as tf:
        tf.extractall(source, filter='data')
    # Upstream HEAD imports an untracked, unrelated module. Disable that unused import.
    init = source / 'common/lora_modules/__init__.py'
    before = init.read_text()
    after = before.replace('from common.lora_modules.rasa_moe import *', '# Reproduction compatibility: unavailable unused rasa_moe import omitted.')
    assert before != after
    init.write_text(after)
    # The repository intentionally does not track the machine's path registry.
    # Every experiment path is passed explicitly, so an empty registry is sufficient.
    (source / 'paths.json').write_text('{}\n')
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile='upstream/common/lora_modules/__init__.py',
                                        tofile='source/common/lora_modules/__init__.py'))
    (HERE / 'compatibility.patch').write_text(patch)
    evidence = HERE / 'evidence'
    evidence.mkdir(exist_ok=True)
    shutil.copy2(source / 'scripts/gora/llama3.1_metamathqa.sh', evidence / 'author_llama3.1_metamathqa.sh')
    shutil.copy2('/workspace/SpecCon-LoRA/reports/gsm8k_speccon_gora_audit_20261116/gora_paper_v3.html', evidence / 'gora_paper_v3.html')
    manifest = {
        'repository': 'https://github.com/hhnqqq/MyTransformers.git',
        'commit': COMMIT,
        'archive_sha256': digest(archive),
        'paper': 'https://arxiv.org/html/2502.12171v3',
        'paper_sha256': digest(evidence / 'gora_paper_v3.html'),
        'compatibility_patch': 'compatibility.patch',
        'files': {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()},
    }
    (HERE / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Frozen {len(manifest["files"])} files at {COMMIT}')

if __name__ == '__main__':
    main()
