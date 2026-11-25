from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import cidfonts, pdfmetrics
from reportlab.platypus import (
    HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = "/SpecCon-LoRA_共识与消融核对报告_20261106.pdf"
pdfmetrics.registerFont(cidfonts.UnicodeCIDFont("STSong-Light"))
font = "STSong-Light"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCN", fontName=font, fontSize=18, leading=25, alignment=TA_CENTER, spaceAfter=9))
styles.add(ParagraphStyle(name="SubCN", fontName=font, fontSize=9, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#526071"), spaceAfter=16))
styles.add(ParagraphStyle(name="HeadCN", fontName=font, fontSize=12, leading=18, textColor=colors.HexColor("#17365D"), spaceBefore=12, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyCN", fontName=font, fontSize=9.5, leading=15.5, spaceAfter=7))
styles.add(ParagraphStyle(name="SmallCN", fontName=font, fontSize=8, leading=12.7, spaceAfter=4))
styles.add(ParagraphStyle(name="CellCN", fontName=font, fontSize=8, leading=12))

story = []
def p(t, style="BodyCN"):
    story.append(Paragraph(t, styles[style]))
def h(t):
    p(t, "HeadCN")
def table(headers, rows, widths):
    data = [[Paragraph(str(x), styles["CellCN"]) for x in headers]] + [
        [Paragraph(str(x), styles["CellCN"]) for x in row] for row in rows
    ]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E9F0F7")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LINEBELOW", (0,0), (-1,0), 0.7, colors.HexColor("#8FA9C2")),
        ("LINEBELOW", (0,-1), (-1,-1), 0.4, colors.HexColor("#D8E1EA")),
    ]))
    story.append(t)
    story.append(Spacer(1, 7))

p("SpecCon-LoRA 论文与共识消融核对报告", "TitleCN")
p("依据 /SpecCon-LoRA.pdf（12 页）与 SpecCon-LoRA 冻结代码、正式结果；核对日期：2026-11-06", "SubCN")
p("<b>总体判断：</b>论文对“跨块左奇异子空间共识”的核心定义、默认预算下的正向结果及几何证据基本匹配实现。最需要处理的是式(8)的归一化分母与代码不一致，以及正文没有交代已完成的跨校准负结果和真实 rank 交换负结果。近期新增的 QQP 三种子配对正结果也尚未写入，但它属于另一套验证集选定的 10-epoch 配方，不能直接并入当前 Table III 而不注明协议。")

h("一、方法定义与实现：匹配处及应修正处")
table(["论文位置", "核对结论", "建议"], [
    ["§III-B 式(4)–(7)", "匹配。代码以左奇异向量前缀的投影重叠计算 C(l,k)，对块对求均值，截断到 C_floor，再以 lambda 次幂乘块均方奇异值。", "可保留。补一句：块谱由带固定种子的随机化截断 SVD 近似得到（oversample=8，power iteration=1）。"],
    ["§III-B 式(8)–(9)", "<b>不完全匹配。</b>论文令 S(l)=前 r_max 个 e(l,j) 之和；冻结 allocator 将前 r_max 项共识修正后，接上聚合梯度的未修正完整谱尾部，再对整个 selection_energy 求和。", "按实际实现改写分母，并注明尾部来自聚合梯度；或者修改代码并重跑实验，不能仅改公式后沿用现有分数。"],
    ["§III-C 式(10)", "论文已说明式(10)是目标表述，实际按可行前缀的成本归一化分数贪心选择。冻结代码另有 strict budget 修补分支。", "明确该算法没有求解式(10)的全局最优；若正式配置启用修补分支，在算法伪代码里写出。"],
    ["§III-D 初始化", "右奇异向量初始化 A、B=0，以及有效 A 倍率 1/sqrt(2)、前向缩放 2，与现有记录一致。", "可保留；区分共识评分使用左子空间，初始化使用聚合梯度右子空间。"],
], [33*mm, 78*mm, 69*mm])
p("<b>建议替换式(8)的文字：</b>“实现中，候选谱的前 r_max 项为共识修正的块均谱能量；超过 r_max 的项沿用聚合梯度的平方奇异值。归一化分母 S_l 为该拼接谱的总和。仅 k≤r_max 的项参与 rank 候选评分。”若保留数学式，可写 S_l = sum_{j=1}^{r_max} e_{l,j} + sum_{j=r_max+1}^{d_l} sigma_bar(l,j)^2，其中 d_l=min(d_in,d_out)。")

h("二、论文已经写对的正向证据")
table(["证据", "当前论文", "核对结果与边界"], [
    ["默认预算同 SVD 初始化", "摘要、§IV-B、Tables I–II", "完整方法减近均匀：CoLA +2.529、MRPC +1.471、RTE +0.481；9 对中 8 胜 1 平。支持整套容量分配，不能全归因于共识。"],
    ["共识组件", "§IV-B、Table III", "参考校准与 SVD 初始化下，完整方法减 Energy-only：+1.632 / +1.716 / +1.805；随机初始化下为 -0.799 / +0.082 / +1.083。论文已写交互，结论应始终限定初始化条件。"],
    ["几何重现", "§IV-D、Table V", "控制 rank、模块类型和能量后，相关为 0.953 / 0.949 / 0.932；这是几何重现预测，不是真实训练收益预测。"],
    ["rank 位置与初始化", "§IV-E、Table VI、附录 F", "同形状 rank 置换 MRPC +0.735、RTE +2.407；同 Gram 的初始化方向对照 +1.528 / +2.042 / +1.324。均已写入。"],
], [43*mm, 42*mm, 95*mm])

h("三、还可补入的正向结果")
table(["优先级", "结果与可用写法", "插入位置 / 限制"], [
    ["高：QQP 新配对", "2026-11-03 重跑兼容的 lambda=0 三种子后，lambda=0.25 的 QQP Accuracy 平均高 <b>+0.0824±0.0264</b> 点，3/3 种子为正；F1 均值高 +0.1207 点。", "可在 §IV-B 新增“补充 QQP 对照”并单列行或附表。该对照为 10 epoch、lambda=0.25；Table IV/VIII 的 QQP 主结果是 15 epoch、lambda=1.5，不能混为同一配置。lambda=0.25 由验证集选择，不能称独立测试收益。"],
    ["中：随机初始化等预算", "完整方法与近均匀组都采用随机初始化时，三任务配对均值为 <b>+0.435 / +0.817 / +0.481</b> 点。", "可在 §IV-B 或附录补一句，说明默认预算下分配收益不只出现在 SVD 初始化条件；它不证明“共识”单独有效。"],
    ["中：低预算", "预算 497,664 参数（严格均匀 rank=3）时，自适应 SVD 减均匀 SVD 为 <b>+1.069 / +0.408 / 0.000</b> 点。", "可在附录预算敏感性表增加一行，连同较大预算 CoLA/MRPC 负收益一起报告；不能概括为所有预算都提升。"],
    ["低：配置诊断", "所测 H=2/4/8/16 中 H=4 三任务均值最高；N=128/256/512/1024 中 N=512 三任务最高。", "可在附录写为配置选择证据。它们不是共识组件本身的独立贡献，也不能称单调或普遍鲁棒。"],
], [21*mm, 84*mm, 75*mm])

h("四、当前论文必须补上的反证与边界")
p("<b>跨校准收益没有复现。</b>原论文在参考校准下报告的三个正均值属实；但替换为五组 A 校准后，固定训练 seed42 的 CoLA/MRPC/RTE 平均差分别为 -0.298 / -0.784 / -0.361 点。MRPC 后续补成 5 校准子集 × 3 训练种子、15 对，平均差仍为 <b>-0.147</b> 点（5 胜、2 平、8 负）。共识 rank 图的跨子集距离也比 Energy-only 大：CoLA 0.1118 vs 0.0962、MRPC 0.0650 vs 0.0598、RTE 0.1970 vs 0.1710。建议至少在 §V Discussion 加一段，避免读者将“几何重现”理解为“分配更稳定或跨校准性能更高”。")
p("<b>真实局部 rank 效用预测为负。</b>MRPC 上预注册 12 个同形状、等预算 rank 交换并重新训练后，共识效用对最终 Accuracy 变化的 Spearman rho=-0.557，方向命中率 25%；Energy utility 为 rho=-0.311、58.3%。这与 §IV-D 的高几何相关并不矛盾，但直接限制“共识分数能预测每个 rank 增量的真实收益”的推论。建议在 §IV-D 或 §V 明示。12 个交换共用参考，属于探索性单任务、单种子结果。")
p("<b>外部基线与统计口径。</b>Table IV 的 DPLoRA 等为跨论文已发表分数，论文已说明协议未统一；保留“表内最高平均值”的描述可以，但不要写成同协议优于 DPLoRA。所有三种子均值是描述性结果。QNLI、QQP 的 lambda 曾在验证集筛选，主表不是未触碰测试集估计。")

h("五、建议直接用于论文的英文补充段落")
p("<b>Discussion:</b> “The consensus term improves mean validation performance under the reference calibration and gradient-SVD initialization, but this benefit does not consistently transfer across alternative calibration draws. On MRPC, 15 paired draw-by-training-seed comparisons yield a mean consensus-minus-energy difference of -0.147 points. Consensus also does not reduce cross-draw rank-map distance in the three tasks examined. Thus, geometric recurrence predictivity should not be interpreted as evidence of calibration-stable rank assignments or guaranteed downstream gains.”", "SmallCN")
p("<b>Rank-utility boundary:</b> “In a separate preregistered MRPC study of 12 feasible budget-preserving rank swaps, consensus utility has a Spearman correlation of -0.557 with the observed final-accuracy changes. This exploratory result indicates that predicting held-out subspace recurrence and predicting the trained utility of a local rank exchange are distinct objectives.”", "SmallCN")
p("<b>QQP supplementary result:</b> “Under a separate 10-epoch QQP protocol, a code-compatible three-seed paired comparison of lambda=0.25 against lambda=0 yields a +0.082±0.026 point mean accuracy difference (3/3 paired wins). Lambda=0.25 was selected on validation data; this comparison is distinct from the 15-epoch, lambda=1.5 QQP configuration in the main eight-task table.”", "SmallCN")

h("六、证据索引")
for s in [
    "原论文：/SpecCon-LoRA.pdf，页 1、3–7、10–11。",
    "冻结实现：/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3/source_snapshot/rte/seed42/speccon_lora/consensus.py 与 rank_allocator.py；特别是 rank_allocator.py 中 _candidate_energy_for、selection_energy.sum()、_base_candidate_score。",
    "默认组件：/workspace/SpecCon-LoRA/experiments/chapter4_core_ablation_v1/component_ablation_report.md；扩展 219 次：/workspace/SpecCon-LoRA/experiments/extended_ablation_20261021_v2/analysis/REPORT.md。",
    "几何复核：/workspace/SpecCon-LoRA/experiments/geometric_predictivity_validation_v1/REPORT.md；MRPC 跨校准：/workspace/SpecCon-LoRA/experiments/mrpc_calibration_robustness_v1/REPORT.md；交换：/workspace/SpecCon-LoRA/experiments/rank_utility_predictivity_v1/REPORT.md。",
    "QQP 新配对：/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1/analysis/QQP_PAIRED.csv；其余配置：/workspace/SpecCon-LoRA/experiments/qnli_qqp_hparam_sweep_20261022_v1/analysis/REPORT.md。",
]: p(s, "SmallCN")

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(font, 8)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(17*mm, 13*mm, "SpecCon-LoRA 论文与共识消融核对报告 · 2026-11-06")
    canvas.drawRightString(195*mm, 13*mm, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=(210*mm, 297*mm), leftMargin=16*mm, rightMargin=16*mm,
                        topMargin=17*mm, bottomMargin=20*mm, title="SpecCon-LoRA 论文与共识消融核对报告")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
