"""Rebuild Chapter 4 tables and figures from completed, immutable run artifacts."""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAIN = ROOT / "experiments/glue_main_ablation_3seeds_gpu12_20261016_v3"
EXT = ROOT / "experiments/extended_ablation_20261021_v2"
SWEEP = ROOT / "experiments/qnli_qqp_hparam_sweep_20261022_v1"
GEO = ROOT / "experiments/geometric_predictivity_validation_v1"
SEEDS = (42, 777, 2026)
TASKS = ("cola", "mrpc", "rte")

def read(p):
    return json.loads(Path(p).read_text())

def csvwrite(name, rows, fields):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def stat(vals):
    return st.mean(vals), st.stdev(vals) if len(vals) > 1 else None

def fmt(vals):
    m, s = stat(vals)
    return f"{m:.3f} $\\pm$ {s:.3f}" if s is not None else f"{m:.3f} (n=1)"

def tex_table(path, caption, label, headers, rows, note=""):
    align = "l" + "r" * (len(headers)-1)
    z = ["\\begin{table*}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    z += [" & ".join(map(str, row)) + " \\\\" for row in rows]
    z += ["\\bottomrule", "\\end{tabular}"]
    if note: z += [f"\\parbox{{0.98\\textwidth}}{{\\footnotesize {note}}}"]
    z += ["\\end{table*}"]
    (OUT / path).write_text("\n".join(z) + "\n")

def metric(p, expected_params=755712):
    d = read(p)
    assert d["state"] == "complete" and d["adapter_parameters"] == expected_params, p
    assert d["selection"].lower().find("final") >= 0, p
    # MRPC's selected primary score is F1 in some runs; the paper table uses Accuracy.
    return float(d["final_accuracy_percent"] if d["task"] == "mrpc" else d["selected_score_percent"])

# Component matrix: original variant names C,D,E,F; near-uniform SVD comes from exact-budget extension.
main_rows = list(csv.DictReader((MAIN / "results.csv").open()))
matrix = {}
source_rows = []
for row in main_rows:
    t, v, seed = row["task"], row["variant"], int(row["seed"])
    if v not in "CDEF": continue
    path = Path(row["metrics"])
    score = metric(path)
    assert abs(score-float(row["score"])) < 1e-8
    matrix[t,v,seed] = score
    source_rows.append(dict(task=t, condition=v, seed=seed, score=score, parameters=755712, source=str(path)))
for t in TASKS:
    for seed in SEEDS:
        p = EXT / "runs" / f"{t}_near_uniform_svd_s{seed}" / "metrics.json"
        matrix[t,"N",seed] = metric(p)
        source_rows.append(dict(task=t, condition="near_uniform_svd", seed=seed, score=matrix[t,"N",seed], parameters=755712, source=str(p)))
assert len(matrix) == 45
csvwrite("component_ablation.csv", source_rows, ["task","condition","seed","score","parameters","source"])
names = {"N":"Near-uniform + Grad-SVD", "D":"Energy-only + Grad-SVD", "E":"Consensus + Random", "F":"Consensus + Grad-SVD", "C":"Energy-only + Random"}
component_rows = []
for v in "NDEFC":
    vals = [[matrix[t,v,s] for s in SEEDS] for t in TASKS]
    component_rows.append([names[v], "755,712"] + [fmt(x) for x in vals])
tex_table("component_ablation.tex", "Exact-budget component ablation on three GLUE tasks (final validation checkpoint; three seeds).", "tab:core_components", ["Configuration","Adapter params","CoLA MCC","MRPC Acc.","RTE Acc."], component_rows, "All configurations use 755,712 adapter parameters. CoLA is MCC $\\times 100$; other scores are accuracy in percent. Random initialization is the original LoRA A random/B zero control. Near-uniform ranks are 4 for attention and 5 for FFN. Mean $\\pm$ sample SD over seeds 42, 777, and 2026.")
contrasts = [("Full - near-uniform","F","N"),("Full - energy-only","F","D"),("Full - random init","F","E"),("Consensus - energy, random init","E","C"),("Grad-SVD - random, energy ranks","D","C")]
cr = []
for label, a, b in contrasts:
    cr.append(dict(contrast=label, **{t:stat([matrix[t,a,s]-matrix[t,b,s] for s in SEEDS])[0] for t in TASKS}))
csvwrite("component_paired_contrasts.csv",cr,["contrast",*TASKS])

# QNLI/QQP: preserve missing and source mismatch as explicit audit states.
five = []
for t, stem in [("qnli","qnli_lr0.000480_e49"),("qqp","qqp_lr0.000440_e10")]:
    for lam in ("0","0.25"):
        for seed in SEEDS:
            key = f"{stem}_lambda{lam}_s{seed}"
            p = SWEEP / "runs" / key
            if not p.exists(): p = SWEEP / "references/results" / key
            if not (p / "metrics.json").exists():
                five.append(dict(task=t,lambda_value=lam,seed=seed,score="",parameters="",source="MISSING",source_sha256=""))
                continue
            m = read(p / "metrics.json"); c = read(p / "run_config.json")
            score = metric(p / "metrics.json")
            five.append(dict(task=t,lambda_value=lam,seed=seed,score=score,parameters=m["adapter_parameters"],source=str(p/"metrics.json"),source_sha256=c["source_sha256"]))
csvwrite("qnli_qqp_consensus_ablation.csv",five,["task","lambda_value","seed","score","parameters","source","source_sha256"])
full_lam = {"cola":"1.5","mrpc":"1.5","rte":"0.5"}
five_table = []
for t in (*TASKS,"qnli","qqp"):
    if t in TASKS:
        energy = [matrix[t,"D",s] for s in SEEDS]
        full = [matrix[t,"F",s] for s in SEEDS]
        lam = full_lam[t]
        status = "matched n=3"
    else:
        a = {lam:{int(r["seed"]):float(r["score"]) for r in five if r["task"]==t and r["lambda_value"]==lam and r["score"]!=""} for lam in ("0","0.25")}
        energy = [a["0"][s] for s in SEEDS if s in a["0"]]
        full = [a["0.25"][s] for s in SEEDS if s in a["0.25"]]
        lam = "0.25"
        status = "matched n=3" if len(energy)==len(full)==3 else "incomplete baseline"
    paired = [full[SEEDS.index(s)]-energy[SEEDS.index(s)] for s in SEEDS] if len(energy)==len(full)==3 else []
    wtl = (f"{sum(x>0 for x in paired)}/{sum(x==0 for x in paired)}/{sum(x<0 for x in paired)}" if paired else "---")
    five_table.append([t.upper() if t!="cola" else "CoLA",lam,fmt(energy) if energy else "---",fmt(full) if full else "---",f"{st.mean(paired):+.3f}" if paired else "---",wtl,status])
tex_table("five_task_consensus_ablation.tex", "Consensus ablation at equal adapter budget across five GLUE tasks.", "tab:five_task_consensus", ["Task","$\\lambda$","Energy-only","Energy + Consensus","Paired $\\Delta$","W/T/L","Status"],five_table,"All completed runs have 755,712 adapter parameters and use Gradient-SVD initialization and final validation checkpoints. CoLA reports MCC $\\times 100$; other tasks report accuracy (percent). The nonzero $\\lambda$ is task-specific. QNLI/QQP $\\lambda=0.25$ was selected after validation sweeps and is not independent test evidence. QQP energy-only seeds 777 and 2026 are missing, so its three-seed paired contrast is deliberately blank.")

# Geometry: frozen EXP-03 table, including the documented grouped-bootstrap CIs.
geo_vals = {"CoLA":("0.204 [0.035, 0.372]","0.967 [0.931, 0.970]","0.953 [0.899, 0.956]","0.001"),"MRPC":("0.148 [-0.055, 0.350]","0.958 [0.903, 0.968]","0.949 [0.889, 0.959]","0.001"),"RTE":("0.284 [0.128, 0.407]","0.954 [0.906, 0.960]","0.932 [0.874, 0.937]","0.001")}
tex_table("geometric_predictivity.tex","Geometric predictivity of cross-block subspace consensus.","tab:geometric_predictivity",["Task","Raw energy","Consensus","Consensus $|$ energy","Permutation $p$"],[[t,*v] for t,v in geo_vals.items()],"Entries are rank correlations with held-out subspace recurrence and 95\\% grouped-bootstrap intervals. The conditional column residualizes log cumulative energy after within-rank/module-type rank normalization. Module-curve permutation uses 999 draws and preserves module type, nested-rank curves, and coarse energy strata. A subsets overlap and share held-out B; these intervals do not describe across-task uncertainty.")
fig, ax = plt.subplots(figsize=(6.7,3.3),layout="constrained")
geo_x=np.arange(3)
for j,(label,color) in enumerate((("Raw energy","#778899"),("Consensus","#32649a"),("Consensus | energy","#d88b42"))):
    values=[]
    for task in ("CoLA","MRPC","RTE"):
        values.append(float(geo_vals[task][j].split()[0]))
    ax.bar(geo_x+(j-1)*.24,values,width=.22,label=label,color=color)
ax.set_xticks(geo_x,["CoLA","MRPC","RTE"])
ax.set_ylabel("Correlation with held-out recurrence")
ax.set_ylim(0,1.1);ax.legend(frameon=False,ncol=3,loc="upper center",fontsize=7)
for ext in ("pdf","svg","png"):fig.savefig(OUT/f"geometric_predictivity_comparison.{ext}",dpi=300)
plt.close(fig)

# Rank placement and initialization controls.
perm = list(csv.DictReader((ROOT/"reports/all_ablation_summary_20261020/04_rank置换/rank置换_逐种子.csv").open(encoding="utf-8-sig")))
placement = []
for t in ("mrpc","rte"):
    x = {kind:[float(next(r["score"] for r in perm if r["task"]==t and r["placement"]==kind and int(r["seed"])==s)) for s in SEEDS] for kind in ("original","permuted")}
    placement.append([t.upper(),fmt(x["original"]),fmt(x["permuted"]),f"{st.mean(a-b for a,b in zip(x['original'],x['permuted'])):+.3f}"])
tex_table("rank_placement_controls.tex","Equal-budget rank placement controls.","tab:rank_placement",["Task","Original","Within-shape permutation","Paired $\\Delta$"],placement,"Both allocations use 755,712 parameters. Permutations preserve the rank multiset within each matrix-shape group; initialization is recomputed from each module's own gradient. Three paired training seeds; final validation accuracy (percent).")
initrows=[]
for label,a,b in [("Full: Grad-SVD vs random","F","E"),("Energy ranks: Grad-SVD vs random","D","C")]:
    initrows.append([label]+[f"{st.mean(matrix[t,a,s]-matrix[t,b,s] for s in SEEDS):+.3f}" for t in TASKS])
pp = read(EXT/"analysis/paired_results.json")
for label, key in [("Full vs matched random orthogonal","direction_same_gram"),("SVD direction, random scale control","svd_scale_only")]:
    vals = []
    for t in TASKS:
        match = next(r for r in pp["contrasts"] if r["task"]==t and r["comparison"]==key)
        assert match["complete"] and match["n"]==3
        vals.append(f"{st.mean(x['difference'] for x in match['paired_values']):+.3f}")
    initrows.append([label]+vals)
tex_table("initialization_controls.tex","Initialization controls (paired mean difference, percentage points).","tab:init_controls",["Contrast","CoLA","MRPC","RTE"],initrows,"The first two rows come from the exact-budget component matrix. Detailed direction and scale controls are reported in the accompanying appendix text; matched orthogonal control preserves Frobenius norm and row Gram structure. All comparisons use final checkpoints and three seeds.")

# Plot formal seed42 rank allocations; compare paired D and F only.
modules = ["Q","K","V","Attn Out","FFN In","FFN Out"]
arrays = {}
summary=[]
for t in TASKS:
    paths = {"energy":MAIN/"references"/t/"seed42"/"reused_D"/"allocation.json", "consensus":MAIN/"references"/t/"seed42"/"reused_F"/"allocation.json"}
    # Some D reference allocations are stored in the main seed directory.
    if not paths["energy"].exists(): paths["energy"] = MAIN/"runs"/t/"D"/"seed42"/"allocation.json"
    if not paths["energy"].exists(): paths["energy"] = MAIN/"references"/t/"seed42"/"D_allocation.json"
    for method,p in paths.items():
        d=read(p); ranks=d["rank_pattern"]
        assert d["rank_report"]["total_adapter_params"]==755712
        mat=np.empty((6,12),dtype=int)
        for name,r in ranks.items():
            m=re.search(r"layer\.(\d+)\.(.*)",name); layer=int(m.group(1)); suffix=m.group(2)
            ix=["attention.self.query","attention.self.key","attention.self.value","attention.output.dense","intermediate.dense","output.dense"].index(suffix)
            mat[ix,layer]=int(r)
            cost=1536 if ix<4 else (768+3072)
            summary.append(dict(task=t,method=method,layer=layer,module=modules[ix],rank=int(r),cost_per_rank=cost,adapter_parameters=int(r)*cost,source=str(p)))
        arrays[t,method]=mat
    assert sum(x["adapter_parameters"] for x in summary if x["task"]==t and x["method"]=="energy")==755712
    assert sum(x["adapter_parameters"] for x in summary if x["task"]==t and x["method"]=="consensus")==755712
csvwrite("rank_allocation_summary.csv",summary,["task","method","layer","module","rank","cost_per_rank","adapter_parameters","source"])
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42})
fig,axes=plt.subplots(3,2,figsize=(10.2,7.2),layout="constrained")
for i,t in enumerate(TASKS):
    for j,method in enumerate(("energy","consensus")):
        ax=axes[i,j]; im=ax.imshow(arrays[t,method],cmap="viridis",vmin=2,vmax=16,aspect="auto")
        ax.set_title(f"{t.upper()} · {method.title()}"); ax.set_xticks(range(12)); ax.set_xticklabels(range(12)); ax.set_yticks(range(6));ax.set_yticklabels(modules if j==0 else [])
        ax.set_xlabel("Transformer layer")
fig.colorbar(im,ax=axes.ravel().tolist(),label="Allocated rank",shrink=.75)
for ext in ("pdf","svg","png"):fig.savefig(OUT/f"rank_allocation_energy_vs_consensus.{ext}",dpi=300)
plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(8.2,7.2),layout="constrained")
shared_diff_limit=max(int(np.abs(arrays[t,"consensus"]-arrays[t,"energy"]).max()) for t in TASKS)
for ax,t in zip(axes,TASKS):
    diff=arrays[t,"consensus"]-arrays[t,"energy"]
    im=ax.imshow(diff,cmap="RdBu_r",vmin=-shared_diff_limit,vmax=shared_diff_limit,aspect="auto")
    ax.set_title(t.upper());ax.set_xticks(range(12));ax.set_xticklabels(range(12));ax.set_yticks(range(6));ax.set_yticklabels(modules);ax.set_xlabel("Transformer layer")
fig.colorbar(im,ax=axes.ravel().tolist(),label="Consensus − energy rank",shrink=.75)
for ext in ("pdf","svg","png"):fig.savefig(OUT/f"rank_allocation_difference.{ext}",dpi=300)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,3.3),layout="constrained")
labels=[]; att=[]; ffn=[]
for t in TASKS:
    for method in ("energy","consensus"):
        vals=[x for x in summary if x["task"]==t and x["method"]==method]
        labels.append(f"{t.upper()}\n{method.title()}")
        att.append(sum(x["adapter_parameters"] for x in vals if x["module"] in modules[:4]))
        ffn.append(sum(x["adapter_parameters"] for x in vals if x["module"] in modules[4:]))
x=np.arange(6);ax.bar(x,att,label="Attention",color="#32649a");ax.bar(x,ffn,bottom=att,label="FFN",color="#d88b42")
ax.set_xticks(x,labels);ax.set_ylabel("Adapter parameters");ax.legend(frameon=False,ncol=2);ax.set_ylim(0,850000)
for ext in ("pdf","svg","png"):fig.savefig(OUT/f"attention_ffn_budget_distribution.{ext}",dpi=300)
plt.close(fig)

lines=["# Rank allocation analysis", "", "Formal seed42 D/F allocations; both use exactly 755,712 adapter parameters. The plots show ranks for all 72 tuned modules and do not imply better downstream accuracy by themselves.", ""]
for t in TASKS:
    e=arrays[t,"energy"];c=arrays[t,"consensus"];d=c-e
    changes = sorted(((int(d[i,j]),j,modules[i]) for i in range(6) for j in range(12)),reverse=True)
    top_up = ", ".join(f"L{j} {m} ({v:+d})" for v,j,m in changes[:3])
    top_down = ", ".join(f"L{j} {m} ({v:+d})" for v,j,m in changes[-3:])
    early = int(d[:,:6].sum()); late = int(d[:,6:].sum())
    lines += [f"## {t.upper()}","",f"Changed modules: {int(np.count_nonzero(d))}/72; increased: {int((d>0).sum())}; decreased: {int((d<0).sum())}.",f"Attention rank change: {int(d[:4].sum()):+d}; FFN rank change: {int(d[4:].sum()):+d}.",f"Attention parameter change: {int(d[:4].sum()*1536):+d}; FFN parameter change: {int(d[4:].sum()*3840):+d}.",f"Early layers 0–5 sum of rank changes: {early:+d}; late layers 6–11: {late:+d}.",f"Largest gains: {top_up}.",f"Largest reductions: {top_down}.",""]
(OUT/"rank_allocation_analysis.md").write_text("\n".join(lines))
print("Built tables, data, and figures in",OUT)
