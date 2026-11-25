"""Independent format, overlap, and token-boundary audit; no GPU allocation."""
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE=Path(__file__).resolve().parent
from launch import TRAIN, DEV, MODEL
from preflight import TEST
from transformers import AutoTokenizer

def main():
    def read(p):
        with open(p) as f:
            return [json.loads(x) for x in f if x.strip()]
    train,dev,test=read(TRAIN),read(DEV),read(TEST)
    train_set={r['input'].strip() for r in train}
    dev_set={r['input'].strip() for r in dev}
    test_set={r['problem'].strip() for r in test}
    overlap=len(train_set & test_set)
    assert overlap==0, f'Exact train/test prompt overlap: {overlap}'
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    results={}
    for label,rows in [('train',train),('dev',dev)]:
        longest=0
        zero_targets=0
        missing_markers=0
        truncated=0
        fully_truncated=0
        max_source=0
        lengths=[]
        assert all(r['input'].startswith('Q: ') and r['input'].endswith('\nA: ') for r in rows)
        for start in range(0,len(rows),512):
            batch=rows[start:start+512]
            sources=tokenizer([r['input'] for r in batch],add_special_tokens=False)['input_ids']
            targets=tokenizer([r['output'] for r in batch],add_special_tokens=False)['input_ids']
            for row,src,tgt in zip(batch,sources,targets):
                total=1+len(src)+len(tgt)+1  # BOS + source + answer + EOS
                longest=max(longest,total)
                max_source=max(max_source,1+len(src))
                truncated+=total>1024
                fully_truncated+=1+len(src)>=1024
                zero_targets+=not bool(tgt)
                missing_markers+='####' not in row['output']
                lengths.append(total)
        assert zero_targets==0 and fully_truncated==0, (label,zero_targets,fully_truncated,max_source)
        results[label]=dict(rows=len(rows),unique_prompts=len({r['input'] for r in rows}),
                            max_sequence_tokens_before_truncation=longest,max_source_tokens=max_source,empty_answers=zero_targets,
                            missing_answer_markers=missing_markers,
                            answer_truncations_at_1024=truncated,fully_truncated_answers=fully_truncated)
    report=dict(status='PASS',data=results,train_dev_exact_prompt_overlap=len(train_set & dev_set),
                train_test_exact_prompt_overlap=overlap,dev_test_exact_prompt_overlap=len(dev_set & test_set),
                prompt='one BOS + raw input(Q/A already included); answer + EOS; no chat template',
                loss_mask='source and padding are -100; answer/EOS contribute to loss',
                limitation='Exact-string test only; no claim that semantic/near-duplicate leakage is excluded')
    (HERE/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
