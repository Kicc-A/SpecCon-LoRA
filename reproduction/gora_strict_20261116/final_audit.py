"""Small final audit of the delivered scripts and actual verification evidence."""
import hashlib
import json
import subprocess
from pathlib import Path
from bootstrap import digest

HERE=Path(__file__).resolve().parent

def main():
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for rel,expected in manifest['files'].items():
        assert digest(HERE/'source'/rel)==expected, rel
    summary=json.loads((HERE/'validation_summary.json').read_text())
    assert {k:v['status'] for k,v in summary.items()}=={'author':'PASS','paper':'PASS'}
    assert not (HERE/'runs').exists() or not any((HERE/'runs').glob('*/final.ckpt')), 'Unexpected full training outputs'
    scripts=sorted(p for p in HERE.iterdir() if p.suffix in ('.py','.sh','.md'))
    for script in scripts:
        if script.suffix=='.sh':
            subprocess.run(['bash','-n',str(script)],check=True)
        elif script.suffix=='.py':
            compile(script.read_text(),str(script),'exec')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(x.split(',')[-1])<512 for x in gpu.strip().splitlines()[:2]), gpu
    audit=dict(status='PASS',frozen_source_files=len(manifest['files']),
               short_run_profiles={k:v['status'] for k,v in summary.items()},
               full_training_started=False,gpu01_released=True,
               delivered_file_sha256={p.name:digest(p) for p in scripts},
               gpu_inventory=gpu)
    (HERE/'final_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='delivered_file_sha256'},indent=2))

if __name__=='__main__':
    main()
