"""LoRA-GA examples/eval_gsm8k.py protocol, using the local GSM8K test cache."""
import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams


def extract_num(text):
    match = re.search(r"####\s*(\d+)", text)
    try:
        return int(match.group(1).replace(',', '')) if match else 0
    except ValueError:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--test', type=Path, default=Path('/workspace/SpecCon-LoRA/reproduction/gora_protocol_20261112/gsm8k_official_1319.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.test.open(encoding='utf-8')]
    assert len(rows) == 1319
    # All GSM8K prompts plus 1024 generated tokens fit; bound KV cache to one GPU.
    model = LLM(str(args.model), dtype='bfloat16', seed=42, max_model_len=1024)
    sampling = SamplingParams(top_p=0.1, top_k=1, temperature=0, max_tokens=512, repetition_penalty=1.0)
    predictions = []
    correct = 0
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        outputs = model.generate([r['problem'] for r in batch], sampling_params=sampling, use_tqdm=False)
        for row, output in zip(batch, outputs):
            answer = output.outputs[0].text
            gt = extract_num(row['answer'])
            pred = extract_num(answer)
            correct += int(gt == pred)
            predictions.append({'prompt': row['problem'], 'reference': row['answer'], 'response': answer, 'gold': gt, 'pred': pred, 'correct': gt == pred})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in predictions), encoding='utf-8')
    result = {'correct': correct, 'total': len(rows), 'accuracy': correct / len(rows), 'eval_seed': 42, 'temperature': 0, 'top_p': 0.1, 'top_k': 1, 'repetition_penalty': 1.0, 'max_tokens': 512, 'batch_size': 4, 'model': str(args.model), 'test': str(args.test)}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
