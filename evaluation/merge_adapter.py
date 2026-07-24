import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-template", default=str(Path(__file__).resolve().parent / "merged_hf"))
    parser.add_argument("--scaling-mode", choices=["rank", "sqrt_rank", "avg_rank"], default="rank")
    args = parser.parse_args()

    adapter = Path(args.adapter)
    output = Path(args.output)
    runtime_template = Path(args.runtime_template)
    config_data = json.loads((adapter / "adapter_config.json").read_text())
    base = Path(config_data["base_model_name_or_path"])

    # This checkpoint was saved with an incorrect SEQ_2_SEQ_LM label. The
    # trained tensors are ordinary Llama causal-LM LoRA tensors, so correct
    # only the in-memory PEFT task label before merging.
    config_data["task_type"] = "CAUSAL_LM"
    config = LoraConfig.from_peft_type(**config_data)

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, adapter, config=config)

    ranks = [int(rank) for rank in config_data.get("rank_pattern", {}).values()]
    avg_rank = sum(ranks) / len(ranks) if ranks else float(config_data["r"])
    alpha = float(config_data["lora_alpha"])
    restored = 0
    for module in model.modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "scaling"):
            continue
        adapter_name = "default"
        rank = module.r[adapter_name] if isinstance(module.r, dict) else module.r
        if args.scaling_mode == "rank":
            scaling = alpha / rank
        elif args.scaling_mode == "sqrt_rank":
            scaling = alpha / math.sqrt(rank)
        else:
            scaling = alpha / math.sqrt(rank * avg_rank)
        if isinstance(module.scaling, dict):
            module.scaling[adapter_name] = scaling
        else:
            module.scaling = scaling
        restored += 1
    if restored == 0:
        raise RuntimeError("No LoRA modules found while restoring scaling")
    print(f"Restored {args.scaling_mode} scaling for {restored} LoRA modules", flush=True)

    model = model.merge_and_unload()
    model.config.torch_dtype = torch.bfloat16
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")

    # Preserve the original Llama tokenizer files for the vLLM 0.6.4 runtime.
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = runtime_template / name
        if source.exists():
            shutil.copy2(source, output / name)

    saved_config = output / "config.json"
    saved = json.loads(saved_config.read_text())
    saved["torch_dtype"] = "bfloat16"
    saved_config.write_text(json.dumps(saved, indent=2) + "\n")


if __name__ == "__main__":
    main()
