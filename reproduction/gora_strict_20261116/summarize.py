"""Aggregate only complete, protocol-matched three-seed GSM8K runs."""
import argparse
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=['author','paper'],default='author')
    a=p.parse_args()
    rows=[]
    protocols=[]
    sources=[]
    for seed in (42,777,2026):
        run=HERE/'runs'/f'{a.profile}-seed{seed}'
        launch=json.loads((run/'launch.json').read_text())
        assert launch['seed']==seed and not launch['smoke'] and launch['profile']==a.profile
        assert json.loads((run/'training_complete.json').read_text())['status']=='PASS'
        assert json.loads((run/'load_merge_validation.json').read_text())['loaded_adapter_tensors']==256
        metrics=json.loads((run/'eval/metrics.json').read_text())
        assert metrics['total']==1319 and abs(metrics['accuracy']-metrics['correct']/1319)<1e-12
        protocol=json.loads((run/'eval/protocol.json').read_text())
        assert protocol['status']=='PASS' and protocol['examples']==1319 and not protocol['smoke']
        protocols.append({k:v for k,v in protocol.items() if k!='model'})
        sources.append((launch['source_commit'],launch['configuration'],launch['reproduction_file_sha256']))
        rows.append(dict(seed=seed,**metrics))
    assert all(x==protocols[0] for x in protocols), 'Evaluation protocols differ'
    assert all(x==sources[0] for x in sources), 'Training source/profile differs'
    values=[r['accuracy']*100 for r in rows]
    report=dict(profile=a.profile,seeds=rows,mean_percent=statistics.mean(values),
                sample_std_percent=statistics.stdev(values),denominator=1319,
                note='Three adapter/training seeds; dataset_seed remains 42 in all runs')
    target=HERE/'runs'/f'{a.profile}-three-seed-summary.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
