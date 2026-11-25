import hashlib,json,shutil,difflib
from pathlib import Path
P=Path('/workspace/SpecCon-LoRA');ORIGIN=P/'experiments/small_glue_rdrop1_rte_mrpc_3seeds_20261003_v1';ROOT=P/'experiments/small_glue_rdrop_dropout_labelsmoothing_3seeds_gpu23_20261008_v1'
old=ORIGIN/'source_snapshot';manifest=json.loads((ORIGIN/'launch_manifest.json').read_text())
for name,h in manifest['source_sha256'].items():assert hashlib.sha256((old/name).read_bytes()).hexdigest()==h,name
ROOT.mkdir(exist_ok=False);(ROOT/'logs').mkdir();snap=ROOT/'source_snapshot';(snap/'reproduction').mkdir(parents=True)
shutil.copytree(old/'speccon_lora',snap/'speccon_lora',ignore=shutil.ignore_patterns('__pycache__'))
source=old/'reproduction/run_roberta_small_glue_rdrop.py';original=source.read_text();s=original
s=s.replace('parser.add_argument("--rdrop-alpha", type=float, default=0.0)','parser.add_argument("--rdrop-alpha", type=float, default=0.0)\n    parser.add_argument("--label-smoothing", type=float, default=0.0)')
s=s.replace('if not math.isfinite(args.rdrop_alpha)', 'if not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 1:\n        parser.error("--label-smoothing must be finite and in [0, 1)")\n    if not math.isfinite(args.rdrop_alpha)')
s=s.replace('def rdrop_loss(first, second, alpha):','def rdrop_loss(first, second, alpha, labels=None, label_smoothing=0.0):\n    first_ce, second_ce = first.loss, second.loss if second is not None else None\n    if label_smoothing > 0:\n        if labels is None:\n            raise ValueError("Label smoothing requires labels")\n        first_ce = torch.nn.functional.cross_entropy(first.logits, labels, label_smoothing=label_smoothing)\n        if second is not None:\n            second_ce = torch.nn.functional.cross_entropy(second.logits, labels, label_smoothing=label_smoothing)')
s=s.replace('return first.loss, first.loss, first.loss.new_zeros(())','return first_ce, first_ce, first_ce.new_zeros(())')
s=s.replace('ce = 0.5 * (first.loss + second.loss)','ce = 0.5 * (first_ce + second_ce)')
s=s.replace('cfg["label_smoothing"] = 0.0','cfg["label_smoothing"] = args.label_smoothing')
s=s.replace('history = []; best = None; global_step = 0;','history = []; best = None; best_accuracy = None; global_step = 0;')
s=s.replace('rdrop_loss(result, second, args.rdrop_alpha)','rdrop_loss(result, second, args.rdrop_alpha, batch["labels"], args.label_smoothing)')
s=s.replace('    final = history[-1]','        if best_accuracy is None or record["splits"]["validation"]["accuracy"] > best_accuracy["splits"]["validation"]["accuracy"]:\n            best_accuracy = dict(record)\n            model.save_pretrained(output / "best_accuracy_adapter")\n            tokenizer.save_pretrained(output / "best_accuracy_adapter")\n    final = history[-1]')
s=s.replace('        "best": best, "best_score_percent": 100 * best["combined_score"],','        "best": best, "best_score_percent": 100 * best["combined_score"],\n        "primary_metric": "validation accuracy x100; fixed final epoch",\n        "final_accuracy_percent": 100 * final["splits"]["validation"]["accuracy"],\n        "best_accuracy": best_accuracy,\n        "best_accuracy_percent": 100 * best_accuracy["splits"]["validation"]["accuracy"],\n        "best_accuracy_adapter": str(output / "best_accuracy_adapter"),')
assert s!=original;compile(s,'run_roberta_small_glue_regularization.py','exec')
(snap/'reproduction/run_roberta_small_glue_regularization.py').write_text(s)
shutil.copy2(source,snap/'reproduction/reference_small_glue_rdrop.py')
(ROOT/'runner_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(source),tofile='run_roberta_small_glue_regularization.py')))
shutil.copy2(Path(__file__),snap/'reproduction')
print(ROOT)
