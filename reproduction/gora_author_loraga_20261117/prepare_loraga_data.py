"""Materialize LoRA-GA examples/data.py load_meta_math using local input files.

The upstream call to dataset.shuffle(seed=42) does not assign its return value,
so the effective selection order is the source dataset order. Preserve that.
"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    samples = json.loads(args.source.read_text(encoding='utf-8'))
    train_path = args.output / 'train100k.jsonl'
    eval_path = args.output / 'eval10k.jsonl'
    seen = kept = rejected_type = rejected_length = 0
    with train_path.open('w', encoding='utf-8') as train, eval_path.open('w', encoding='utf-8') as dev:
        for sample in samples:
            seen += 1
            row = {
                'input': f'Q: {sample["query"]}\nA: ',
                'output': sample['response'].split('\nThe answer is:')[0],
            }
            if 'GSM' not in sample['type']:
                rejected_type += 1
                continue
            if len(tokenizer(row['input'] + ' ' + row['output'])['input_ids']) >= 512:
                rejected_length += 1
                continue
            if kept < 100000:
                train.write(json.dumps(row, ensure_ascii=False) + '\n')
            elif kept < 110000:
                dev.write(json.dumps(row, ensure_ascii=False) + '\n')
            else:
                break
            kept += 1
            if kept == 110000:
                break
    assert kept == 110000, (seen, kept)
    manifest = {
        'protocol': 'LoRA-GA examples/data.py load_meta_math effective code path',
        'source': str(args.source), 'source_sha256': digest(args.source),
        'tokenizer': str(args.tokenizer),
        'selection': 'source order; type contains GSM; tokenized prompt+space+answer < 512; first 100000 train, next 10000 eval',
        'prompt': 'Q: {query}\\nA: ',
        'answer': 'response split before first \\nThe answer is:',
        'seen': seen, 'kept': kept,
        'rejected_type': rejected_type, 'rejected_length': rejected_length,
        'train_sha256': digest(train_path), 'eval_sha256': digest(eval_path),
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
