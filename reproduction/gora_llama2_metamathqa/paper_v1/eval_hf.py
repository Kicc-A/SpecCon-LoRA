"""LoRA-GA reproduce/ generation and scoring functions, with local data paths."""
import json
import re
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def model_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_text: str,
    model_type: str,
    max_source_length: str = 768,
    max_target_length: str = 256,
):
    if model_type == "CausalLM":
        inputs = tokenizer(
            input_text + " ",
            return_tensors="pt",
            max_length=max_source_length,
            truncation=True,
            return_token_type_ids=False,
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_target_length,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                temperature=0.8,
            )
        pred_text = tokenizer.decode(
            outputs.sequences[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=True,
        )
    elif model_type == "ConditionalGeneration":
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_target_length)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pred_text

def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r'####\s*(\d+)'
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
    else:
        print(text)
        result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0

def main(run, seed):
    model_dir = run / "merged_hf"
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    print("Model loaded successfully (SDPA)", flush=True)
    test = Path("/workspace/SpecCon-LoRA/reproduction/gora_protocol_20261112/gsm8k_official_1319.jsonl")
    rows = [json.loads(line) for line in test.read_text().splitlines()]
    assert len(rows) == 1319 and all(row["problem"].endswith("\nA: ") for row in rows)
    correct = 0
    with (run / "predictions.jsonl").open("x") as out:
        for row in rows:
            response = model_inference(model, tokenizer, row["problem"], "CausalLM", max_target_length=512)
            ok = extract_num(response) == extract_num(row["answer"])
            correct += int(ok)
            print(json.dumps(dict(row, response=response, correct=ok)), file=out, flush=True)
    result = dict(train_seed=int(run.name.split("seed")[-1]), eval_seed=seed, correct=correct, total=len(rows), accuracy=correct/len(rows))
    (run / "result.json").write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
