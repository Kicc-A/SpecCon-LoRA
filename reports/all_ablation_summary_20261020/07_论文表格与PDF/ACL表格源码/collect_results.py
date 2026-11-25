"""Freeze auditable result inputs without touching running experiments."""
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = Path('/workspace/SpecCon-LoRA')
EXP = ROOT / 'experiments'
TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']
SEEDS = [42, 777, 2026]
MAIN = EXP / 'glue_main_ablation_3seeds_gpu12_20261016_v3'
RDROP = EXP / 'glue_rdrop_ablation_seed42_gpu3_20261016_v2'
ZERO = EXP / 'glue_lambda0_seed42_gpu2_20261015_v1'
L15 = EXP / 'rte_qnli_lambda15_3seeds_gpu012_20261015_v1'
L07 = EXP / 'qnli_lambda07_3seeds_gpu0_20261016_v1'
EXTRA = EXP / 'glue_paper_extras_gpu3_20261016_v1'
CACHE, SOURCES = {}, {}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def read(path):
    path = Path(path)
    key = str(path)
    if key not in CACHE:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        target = OUT / 'sources' / f'{sha[:16]}_{path.name}'
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(raw)
        SOURCES[key] = {'sha256': sha, 'snapshot': str(target.relative_to(OUT))}
        CACHE[key] = json.loads(raw) if path.suffix == '.json' else raw.decode('utf-8-sig')
    return CACHE[key]


def score(task, metric):
    splits = metric['final']['splits']
    if task == 'mnli':
        return 50 * (splits['matched']['accuracy'] + splits['mismatched']['accuracy'])
    field = {'cola': 'matthews_correlation', 'stsb': 'pearson'}.get(task, 'accuracy')
    return 100 * splits['validation'][field]


def run(path, task, seed):
    path = Path(path)
    if not path.is_file():
        return None
    m = read(path)
    if m.get('state') != 'complete':
        return None
    assert m['task'] == task and m['seed'] == seed, path
    assert m['epochs'] == m['planned_epochs'] and not m['stopped_early'], path
    assert not any(x in path.parts for x in ['preflight']), path
    return {'task': task, 'seed': seed, 'score': score(task, m),
            'epochs': m['epochs'], 'parameters': m['adapter_parameters'],
            'elapsed_seconds': m.get('elapsed_seconds'), 'hyperparameters': m['hyperparameters'],
            'metrics': str(path), 'sha256': SOURCES[str(path)]['sha256']}


def main():
    started = now()
    report = read(ROOT / 'reports/glue_best_42_777_2026_20261013/results.json')
    selected = {}
    for task, group in report['tasks'].items():
        selected[task] = []
        for row in group['runs']:
            item = run(ROOT / row['source'] / 'metrics.json', task, row['seed'])
            assert item and abs(item['score'] - row['score']) < 1e-8
            selected[task].append(item)

    manifest = read(MAIN / 'manifest.json')
    refs = {x['id']: x for x in manifest['references']}
    components = []
    for cell in manifest['cells']:
        ref = refs.get(cell['id'])
        path = Path(ref['metrics']) if ref else Path(cell['output']) / 'metrics.json'
        item = run(path, cell['task'], cell['seed'])
        if item:
            assert item['parameters'] == cell['adapter_parameters']
        components.append({'task': cell['task'], 'seed': cell['seed'], 'variant': cell['variant'],
                           'parameters': cell['adapter_parameters'], 'result': item,
                           'reused': bool(ref), 'original_source': ref['source'] if ref else cell['output']})

    rdrop = []
    for cell in read(RDROP / 'results.json')['cells']:
        item = run(cell['metrics'], cell['task'], cell['seed']) if cell.get('metrics') else None
        assert item and abs(item['score'] - cell['score']) < 1e-8
        rdrop.append({**{k: cell[k] for k in ['task', 'seed', 'adapter_method', 'rdrop_alpha']}, 'result': item})

    zeros = {task: run(ZERO / 'runs' / task / 'seed42/metrics.json', task, 42) for task in TASKS}
    sensitivity = []
    for task in ['rte', 'qnli']:
        original_lambda = report['tasks'][task]['hyperparameters']['lambda']
        sensitivity.append({'task': task, 'lambda': original_lambda, 'results': selected[task]})
        if task == 'rte':
            sensitivity.append({'task': task, 'lambda': 0, 'results': [x['result'] for x in components if x['task'] == 'rte' and x['variant'] == 'D' and x['result'] is not None]})
        for lam, directory in [(1.5, L15)] + ([(0.7, L07)] if task == 'qnli' else []):
            items = [run(directory / 'runs' / task / f'seed{s}/metrics.json', task, s) for s in SEEDS]
            sensitivity.append({'task': task, 'lambda': lam, 'results': [x for x in items if x]})

    extra_manifest = read(EXTRA / 'manifest.json')
    extended_rdrop = []
    for cell in extra_manifest['cells']:
        item = run(Path(cell['source']) / 'metrics.json', cell['task'], cell['seed'])
        extended_rdrop.append({'task': cell['task'], 'seed': cell['seed'],
                               'method': cell['adapter_method'], 'alpha': cell['rdrop_alpha'], 'result': item})
    shuffled = []
    for job in extra_manifest['jobs']:
        if 'rankshuffle' in job['id']:
            item = run(Path(job['output']) / 'metrics.json', job['task'], job['seed'])
            shuffled.append({'task': job['task'], 'seed': job['seed'], 'result': item})

    dp = read('/workspace/DPLoRA/experiments/glue_dp04_full_20260922/completed_runs.json')
    dp_runs = []
    for key, row in dp.items():
        evaluation = read(row['metric_file'])
        training = read(Path(row['output_dir']) / 'train_metrics.json')
        pruning = read(Path(row['output_dir']) / 'pruning/pruning_metrics.json')
        expected = row['score_percent'] / 100
        numbers = [v for v in evaluation.values() if isinstance(v, (float, int))]
        assert any(abs(v - expected) < 1e-8 for v in numbers), (key, evaluation)
        dp_runs.append({'task': key.split('/')[0], 'seed': row['seed'], 'score': row['score_percent'],
                        'epochs': training['epoch'], 'parameters': pruning['final_parameters'],
                        'metrics': row['metric_file']})

    math_path = EXP / 'llama31_metamathqa_specconlora_fair_rmin6_rmax24_gamma0.7_lambda0.5_seed77_gpu0123/eval/gsm8k_vllm_loraga/result/metrics_loraga.json'
    code_path = EXP / 'llama31_codefeedback_specconlora_metaconfig_rmin6_rmax24_gamma0.7_lambda0.5_seed42_gpu0123/eval/humaneval_official.jsonl.metrics.json'
    math, code = read(math_path), read(code_path)
    assert abs(math['accuracy'] - math['correct'] / math['total']) < 1e-10
    mtroot = ROOT / 'reports/mtbench_seed42_speccon_vs_dplora_20261014'
    mt = list(csv.DictReader(io.StringIO(read(mtroot / 'score_comparison.csv'))))
    read(mtroot / 'export_manifest.json')
    read(mtroot / 'rank_diagnosis.json')
    read(EXP / 'mtbench_llama32_1b_speccon_author_protocol_20261013_v1/aggregate_scores.json')

    campaigns = {}
    for name, directory in [('lambda0', ZERO), ('lambda15', L15), ('lambda07', L07),
                            ('components', MAIN), ('rdrop42', RDROP), ('extras', EXTRA)]:
        campaigns[name] = read(directory / 'status.json')
        if (directory / 'README.md').exists(): read(directory / 'README.md')
    exclusions = {}
    for directory in ['glue_main_ablation_3seeds_gpu12_20261016_v1',
                      'glue_main_ablation_3seeds_gpu12_20261016_v2',
                      'glue_rdrop_ablation_seed42_gpu3_20261016_v1']:
        exclusions[directory] = read(EXP / directory / 'status.json')
    snapshot = {'snapshot_started': started, 'snapshot_finished': now(),
                'selected_report': report, 'selected': selected, 'components': components,
                'rdrop42': rdrop, 'lambda0': zeros, 'sensitivity': sensitivity,
                'extended_rdrop': extended_rdrop, 'rank_shuffle': shuffled,
                'dplora_local': dp_runs, 'math': math, 'code': code, 'mtbench': mt,
                'campaign_status': campaigns, 'excluded_versions': exclusions,
                'source_files': SOURCES}
    (OUT / 'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'snapshot': snapshot['snapshot_finished'], 'source_files': len(SOURCES),
                      'main_runs': sum(map(len, selected.values())),
                      'component_cells_complete': sum(x['result'] is not None for x in components),
                      'lambda0_complete': sum(x is not None for x in zeros.values()),
                      'rdrop42_cells': len(rdrop), 'extended_rdrop_cells_complete': sum(x['result'] is not None for x in extended_rdrop),
                      'shuffle_complete': sum(x['result'] is not None for x in shuffled)}, indent=2))


if __name__ == '__main__':
    main()
