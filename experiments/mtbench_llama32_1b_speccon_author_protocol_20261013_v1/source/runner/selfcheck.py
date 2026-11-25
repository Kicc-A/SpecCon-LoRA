"""CPU integration checks, with tiny random weights and offline historical ratings."""
import argparse
import ast
from dataclasses import replace
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from common import AUTHOR_ROOT, check_environment, read_json, rows, summarize_judgments, write_json
from train import AlpacaCollator, author_backward, nonquantized_peft, optimizer_for


def main(report_path):
    if torch.cuda.is_available():
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' so this check never uses an occupied GPU")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    sys.path.insert(0, "/workspace/SpecCon-LoRA")
    from speccon_lora import SpecConLoraConfig, prepare_specconlora_model
    versions = check_environment(AUTHOR_ROOT / "requirements_alpaca.txt")
    cfg = read_json(Path(__file__).with_name("config_1b.json"))
    gs = replace(SpecConLoraConfig(**cfg["speccon"]), param_budget=3072, r_max=8,
                 calibration_steps=4, calibration_blocks=2)
    tiny_config = LlamaConfig(vocab_size=96, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    tiny = LlamaForCausalLM(tiny_config)
    batches = []
    for _ in range(4):
        ids = torch.randint(3, 96, (2, 16))
        labels = ids.clone()
        labels[:, :6] = -100
        batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    with tempfile.TemporaryDirectory(prefix="speccon_mtbench_check_") as tmp:
        tmp = Path(tmp)
        tiny.save_pretrained(tmp / "base")
        tiny = AutoModelForCausalLM.from_pretrained(tmp / "base")
        tiny.gradient_checkpointing_enable()
        tiny.enable_input_require_grads()
        tiny.config.use_cache = False
        with nonquantized_peft(tiny):
            model, report = prepare_specconlora_model(tiny, batches, lambda m, b: m(**b).loss, gs)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3072
        assert all(2 <= r <= 8 for r in report["rank_pattern"].values())
        assert len(report["target_names"]) == 14
        model.peft_config["default"].use_rslora = True
        opt = optimizer_for(model, cfg)
        acc = Accelerator(cpu=True, gradient_accumulation_steps=8)
        # Confirm preservation of the original loop's actual gradient scaling.
        scalar = torch.tensor(2.0, requires_grad=True)
        author_backward(acc, scalar.square(), 8)
        assert scalar.grad.item() == 4 / 64
        start = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for batch in batches:
            loss = model(**batch).loss
            assert torch.isfinite(loss)
            author_backward(acc, loss, 8)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        opt.step()
        opt.zero_grad()
        changed = sum(not torch.equal(start[n], p) for n, p in model.named_parameters() if n in start)
        assert changed > 0
        model.eval()
        with torch.no_grad():
            before = model(**batches[0]).logits
        model.save_pretrained(tmp / "adapter")
        fresh = AutoModelForCausalLM.from_pretrained(tmp / "base")
        with nonquantized_peft(fresh):
            loaded = PeftModel.from_pretrained(fresh, tmp / "adapter").eval()
        with torch.no_grad():
            reloaded = loaded(**batches[0]).logits
        torch.testing.assert_close(before, reloaded, atol=1e-6, rtol=1e-5)
        merged = loaded.merge_and_unload(safe_merge=True).eval()
        with torch.no_grad():
            after = merged(**batches[0]).logits
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)
        merged.save_pretrained(tmp / "final_model")
        exported = AutoModelForCausalLM.from_pretrained(tmp / "final_model").eval()
        with torch.no_grad():
            disk_logits = exported(**batches[0]).logits
        torch.testing.assert_close(after, disk_logits, atol=1e-6, rtol=1e-5)
        merge_error = float((before - after).abs().max())

    # Compare the collator against the exact author class without importing its trainer.
    tree = ast.parse((AUTHOR_ROOT / "training/train_alpaca.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_AlpacaCollator")
    namespace = {"torch": torch, "pad_sequence": torch.nn.utils.rnn.pad_sequence}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "author_collator", "exec"), namespace)
    tokenizer = SimpleNamespace(pad_token_id=0)
    inputs = [{"input_ids": [1, 5, 7], "labels": [-100, 5, 7]},
              {"input_ids": [1, 6], "labels": [-100, 6]}]
    ours = AlpacaCollator(tokenizer)(inputs)
    theirs = namespace["_AlpacaCollator"](tokenizer)(inputs)
    assert all(torch.equal(ours[k], theirs[k]) for k in ours)

    # Exercise the actual author judge prompt builder offline on all historical turns.
    sys.path.insert(0, str(AUTHOR_ROOT / "FastChat"))
    from fastchat.conversation import get_conv_template
    judge_root = AUTHOR_ROOT / "FastChat/fastchat/llm_judge"
    tree = ast.parse((judge_root / "common.py").read_text())
    selected = [n for n in tree.body if
                (isinstance(n, ast.FunctionDef) and n.name == "run_judge_single") or
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                 {"one_score_pattern", "one_score_pattern_backup"} for t in n.targets))]
    capture = {}

    def offline_completion(model, conv, temperature, max_tokens):
        assert (model, temperature, max_tokens) == ("gpt-4", 0, 2048)
        return capture["judgment"]

    namespace = {"ast": ast, "re": re, "OPENAI_MODEL_LIST": ["gpt-4"], "ANTHROPIC_MODEL_LIST": [],
                 "get_conversation_template": lambda _: get_conv_template("chatgpt"),
                 "chat_completion_openai": offline_completion}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "author_judge", "exec"), namespace)
    previous = AUTHOR_ROOT / "experiments/mtbench_llama32_1b_dp06_seed42_paper_20261011_v4"
    answer_path = previous / "mt_bench_answers_seed42_paper.jsonl"
    judgment_path = previous / "full_gpt4_judgment/gpt4_judgments_160.jsonl"
    answers = {r["question_id"]: r for r in rows(answer_path)}
    questions_path = judge_root / "data/mt_bench/question.jsonl"
    questions = {r["question_id"]: r for r in rows(questions_path)}
    refs = {r["question_id"]: r for r in rows(judge_root / "data/mt_bench/reference_answer/gpt-4.jsonl")}
    prompts = {r["name"]: r for r in rows(judge_root / "data/judge_prompts.jsonl")}
    checked = 0
    for rating in rows(judgment_path):
        qid, turn = rating["question_id"], rating["turn"]
        reference = questions[qid]["category"] in {"math", "reasoning", "coding"}
        template = ("single-math-v1" if reference else "single-v1") + ("-multi-turn" if turn == 2 else "")
        capture["judgment"] = rating["judgment"]
        judge = SimpleNamespace(model_name="gpt-4", prompt_template=prompts[template])
        score, prompt, _ = namespace["run_judge_single"](
            questions[qid], answers[qid], judge, refs[qid] if reference else None, multi_turn=turn == 2)
        assert prompt == rating["user_prompt"] and score == rating["score"]
        checked += 1
    assert checked == 160
    historical = summarize_judgments(judgment_path, questions_path, next(iter(answers.values()))["model_id"])
    assert historical["score"] == 3.615625
    result = {"passed": True, "device": "CPU", "formal_training_started": False, "api_calls": 0,
              "author_pins_checked": len(versions), "tiny_model_target_layers": 14,
              "tiny_model_trainable_parameters": 3072, "updated_adapter_tensors": changed,
              "adapter_reload_logits_match": True, "merged_logits_max_abs_error": merge_error,
              "merged_export_reload_match": True, "collator_matches_author": True,
              "author_gradient_accumulation_verified": True, "historical_prompts_exact": checked,
              "historical_score_recomputed": historical["score"],
              "limits": ["Tiny FP32 CPU check; full 1B BF16 GPU memory/runtime not measured.",
                         "Historical judgments used for offline protocol verification; no SpecCon score yet."]}
    write_json(report_path, result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    main(parser.parse_args().report)
