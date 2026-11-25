import json
import re
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:3])
rows = [json.loads(line) for line in source.open()]
pattern = re.compile(r"####\s*(\d+)")


def extract(text):
    match = pattern.search(text)
    return int(match.group(1)) if match else 0


correct = sum(extract(row['test_result']) == extract(row['answer']) for row in rows)
missing = sum(pattern.search(row['test_result']) is None for row in rows)
metrics = {'accuracy': correct / len(rows), 'correct': correct, 'total': len(rows),
           'missing_pred_marker': missing,
           'extractor': 'LoRA-GA GSM8K: #### positive integer; missing -> 0'}
target.write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps(metrics, indent=2))
