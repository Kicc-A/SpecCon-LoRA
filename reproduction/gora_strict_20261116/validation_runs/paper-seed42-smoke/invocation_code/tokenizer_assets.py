"""Keep the original pinned tokenizer assets compatible with both runtime environments."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

def export_tokenizer(source, target):
    source,target=Path(source),Path(target)
    copied={}
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','tokenizer.model'):
        origin=source/name
        if origin.exists():
            shutil.copyfile(origin,target/name)
            copied[name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
            assert (target/name).read_bytes()==origin.read_bytes()
    assert 'tokenizer.json' in copied and 'tokenizer_config.json' in copied
    return dict(strategy='Original pinned tokenizer files copied byte-for-byte; no Transformers5 TokenizersBackend class serialization',
                sha256=copied)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    config=json.loads((a.run/'config.json').read_text())
    report=json.loads((a.run/'load_merge_validation.json').read_text())
    report['tokenizer_export']=export_tokenizer(config['model_name_or_path'],a.run/'merged_hf')
    (a.run/'load_merge_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['tokenizer_export'],indent=2))
