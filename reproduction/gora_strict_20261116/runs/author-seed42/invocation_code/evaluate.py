"""Pinned original decoding defaults; no test-set tuning, raw Q/A prompts."""
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata as md
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = Path('/workspace/SpecCon-LoRA/reproduction/gora_protocol_20261112/gsm8k_official_1319.jsonl')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', choices=['author','paper'], default='author')
    p.add_argument('--seed', type=int, default=42, help='Training seed; decoding seed is fixed at 42')
    p.add_argument('--smoke', action='store_true', help='Three synthetic prompts, no GSM8K test scoring')
    a = p.parse_args()
    name = f'{a.profile}-seed{a.seed}' + ('-smoke' if a.smoke else '')
    run = HERE / ('validation_runs' if a.smoke else 'runs') / name
    validation = json.loads((run / 'load_merge_validation.json').read_text())
    assert validation['status'] == 'PASS' and validation['loaded_adapter_tensors'] == 256
    export = run / 'merged_hf'
    assert export.is_dir()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    if a.smoke:
        rows = [{'problem': x} for x in [
            'Q: What is 13 plus 29?\nA: ',
            'Q: A box holds 7 apples. How many apples do 3 boxes hold?\nA: ',
            'Q: There are 30 pencils. Five are given away. How many remain?\nA: ']]
    else:
        expected = json.loads((HERE / 'inputs.lock.json').read_text())[str(TEST)]['sha256']
        assert hashlib.sha256(TEST.read_bytes()).hexdigest() == expected
        rows = [json.loads(line) for line in TEST.read_text().splitlines() if line.strip()]
        assert len(rows) == 1319 and len({r['problem'] for r in rows}) == 1319
    target = run / ('engine_smoke' if a.smoke else 'eval')
    target.mkdir(exist_ok=False)
    with (HERE / 'gpu01.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True)
        tokens = [tokenizer(r['problem'], add_special_tokens=True).input_ids for r in rows]
        assert max(map(len,tokens)) + (64 if a.smoke else 512) <= 1024
        expected_first_tokens = None
        if a.smoke:
            # Independent HF forward of the exported model, before vLLM reserves GPUs.
            import torch
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(export, torch_dtype=torch.bfloat16,
                                                      attn_implementation='sdpa', local_files_only=True).eval().to('cuda:0')
            expected_first_tokens = []
            with torch.no_grad():
                for ids in tokens:
                    batch = torch.tensor([ids],device='cuda:0')
                    logits = hf(input_ids=batch,use_cache=False).logits[0,-1].float()
                    # vLLM applies repetition_penalty to prompt as well as generated
                    # tokens. Compare the same decoding rule, not raw-logit argmax.
                    prompt_ids = torch.tensor(sorted(set(ids)),device='cuda:0')
                    values = logits[prompt_ids]
                    logits[prompt_ids] = torch.where(values>0,values/1.1,values*1.1)
                    expected_first_tokens.append(logits.argmax().item())
            del hf, batch
            gc.collect()
            torch.cuda.empty_cache()
        llm = LLM(model=str(export), tokenizer=str(export), tensor_parallel_size=2,
                  dtype='bfloat16', max_model_len=1024, seed=42, gpu_memory_utilization=.85,
                  disable_log_stats=True)
        params = SamplingParams(temperature=0, top_p=.1, max_tokens=64 if a.smoke else 512,
                                seed=42, repetition_penalty=1.1, stop='</answer>',
                                include_stop_str_in_output=True)
        outputs = llm.generate(prompt_token_ids=tokens, sampling_params=params)
        assert len(outputs) == len(rows)
        for row, output in zip(rows,outputs):
            row['test_result'] = output.outputs[0].text
            row['generated_token_ids'] = list(output.outputs[0].token_ids)
            assert row['test_result'] and row['generated_token_ids']
        actual_first_tokens = [r['generated_token_ids'][0] for r in rows]
        if a.smoke:
            assert actual_first_tokens == expected_first_tokens, (expected_first_tokens,actual_first_tokens)
        (target / 'generations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        protocol = dict(status='PASS', smoke=a.smoke, examples=len(rows), model=str(export),
                        gpus=[0,1], tensor_parallel_size=2, vllm=md.version('vllm'),
                        prompt='raw problem field, tokenizer BOS, no chat/instruction/system template',
                        temperature=0, top_p=.1, repetition_penalty=1.1, seed=42,
                        max_tokens=64 if a.smoke else 512, max_model_len=1024,
                        top_k='upstream argument unused; vLLM default preserved',
                        hf_vllm_same_first_tokens_on_synthetic_prompts=(actual_first_tokens==expected_first_tokens) if a.smoke else None,
                        hf_reference_first_token='HF logits with same prompt repetition_penalty1.1 as vLLM',
                        hf_first_tokens=expected_first_tokens,
                        caveat='Paper does not pin all decoding settings; follows repository defaults and existing local GSM8K protocol')
        if not a.smoke:
            pattern = re.compile(r'####\s*(\d+)')
            def extract(s):
                m = pattern.search(s)
                return int(m.group(1)) if m else 0
            correct = sum(extract(r['test_result']) == extract(r['answer']) for r in rows)
            metrics = dict(correct=correct,total=len(rows),accuracy=correct/len(rows),
                           missing_pred_marker=sum(pattern.search(r['test_result']) is None for r in rows),
                           extractor='LoRA-GA GSM8K: first #### positive integer; missing -> 0')
            robust = re.compile(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)')
            def parse(s):
                match = robust.search(s)
                return match.group(1).replace(',','').lstrip('+') if match else None
            diagnostic = sum(parse(r['test_result']) is not None and parse(r['test_result']) == parse(r['answer']) for r in rows)
            metrics['diagnostic_signed_comma_correct'] = diagnostic
            (target / 'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
            print(json.dumps(metrics,indent=2))
        (target / 'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (target / 'evaluate.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(protocol,indent=2))

if __name__ == '__main__':
    main()
