"""Create the dated Chinese PDF summary from verified ablation results."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import cidfonts, pdfmetrics
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path('/workspace/SpecCon-LoRA/reports/SpecCon-LoRA_消融实验阶段报告_20261107.pdf')
pdfmetrics.registerFont(cidfonts.UnicodeCIDFont('STSong-Light'))
FONT = 'STSong-Light'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CTitle', fontName=FONT, fontSize=17, leading=23,
                          alignment=TA_CENTER, spaceAfter=9))
styles.add(ParagraphStyle(name='CSub', fontName=FONT, fontSize=9, leading=13,
                          alignment=TA_CENTER, textColor=colors.HexColor('#526071'), spaceAfter=12))
styles.add(ParagraphStyle(name='CHead', fontName=FONT, fontSize=12, leading=17,
                          textColor=colors.HexColor('#153757'), spaceBefore=12,
                          spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle(name='CBody', fontName=FONT, fontSize=9.2, leading=15,
                          spaceAfter=6))
styles.add(ParagraphStyle(name='CCell', fontName=FONT, fontSize=8.1, leading=12))
styles.add(ParagraphStyle(name='CSmall', fontName=FONT, fontSize=8.1, leading=12.5,
                          spaceAfter=5))
story = []


def p(text, style='CBody'):
    story.append(Paragraph(text, styles[style]))


def h(text):
    p(text, 'CHead')


def table(headers, rows, widths):
    data = [[Paragraph(str(v), styles['CCell']) for v in headers]]
    data += [[Paragraph(str(v), styles['CCell']) for v in row] for row in rows]
    item = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    item.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8eff6')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 5), ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,0), (-1,0), 0.6, colors.HexColor('#829cb5')),
        ('LINEBELOW', (0,-1), (-1,-1), 0.3, colors.HexColor('#d8e0e8')),
    ]))
    story.append(item)
    story.append(Spacer(1, 5))


p('SpecCon-LoRA 消融实验阶段报告', 'CTitle')
p('更新日期：2026-11-07 21:38（北京时间） · 对照论文 /SpecCon-LoRA.pdf', 'CSub')
p('<b>核心判断：</b>默认预算下整套自适应 rank 分配有效；参考校准与 SVD 初始化下共识组件为正。新增两组共识错配和左／右／联合评分实验进一步支持共识对应关系及左侧子空间选择，但跨校准负结果和局部 rank 效用负结果仍需在论文中并列报告。')

h('1. 原有核心组件证据')
table(['完整方法减控制组', 'CoLA', 'MRPC', 'RTE', '说明'], [
    ['同预算近均匀 + SVD', '+2.529', '+1.471', '+0.481', '九配对八胜一平；整套分配收益'],
    ['Energy-only + SVD', '+1.632', '+1.716', '+1.805', '参考校准下共识组件收益'],
    ['共识分配 + 随机初始化', '+2.186', '+1.307', '+1.324', 'SVD 初始化贡献'],
], [52*mm, 19*mm, 19*mm, 19*mm, 65*mm])
p('三任务均为 755,712 个适配器参数、三训练种子、固定末轮验证指标。CoLA 为 MCC×100，MRPC/RTE 为 Accuracy×100。QNLI 验证集选定 λ=0.25 的共识增益为 +0.214 点；QQP 10-epoch 配方下 λ=0.25 减 λ=0 为 +0.082±0.026 点，不能与 15-epoch 主表配置混用。', 'CSmall')

h('2. 共识曲线错配：Full 减 Shuffle')
table(['任务', '第一组预注册 shuffle', '第二组预注册 shuffle'], [
    ['CoLA', '+2.543±1.287（3/0/0）', '+1.649±2.072（2/1/0）'],
    ['MRPC', '+2.124±0.861（3/0/0）', '+0.327±0.566（1/2/0）'],
    ['RTE', '+1.444±1.083（3/0/0）', '未运行'],
], [30*mm, 72*mm, 72*mm])
p('括号为胜/平/负。实验在同形状、同模块类型内错配整条共识曲线，保留模块自身谱能量，随后重新分配并完整训练。第二组 MRPC 收益明显更小；两组结果均须报告。两组 shuffle 复用相同训练种子，不构成六个独立训练种子。', 'CSmall')

h('3. 子空间评分选择：三种子配对差值')
table(['任务', '左侧-右侧', '左侧-联合', '联合-右侧'], [
    ['CoLA', '+2.185±1.051（3/0/0）', '+0.313±0.645（2/0/1）', '+1.873±1.241（3/0/0）'],
    ['MRPC', '+1.471±0.490（3/0/0）', '+0.899±1.020（2/0/1）', '+0.572±1.020（2/0/1）'],
    ['RTE', '+1.805±0.625（3/0/0）', '+1.444±2.527（2/0/1）', '+0.361±1.910（2/0/1）'],
], [24*mm, 51*mm, 51*mm, 48*mm])
p('联合评分为每个 rank 前缀的 sqrt(C_left × C_right)。右侧与联合沿用相同随机化 SVD 的谱能量、预算、初始化和任务训练配方。左侧相对右侧在三任务九个配对中全部为正；相对联合每任务有一个负配对，不能写成逐种子稳定优于联合。', 'CSmall')

story.append(PageBreak())
h('4. 论文必须保留的限制')
p('（1）MRPC 五校准子集×三训练 seed 的 15 对，完整共识减 Energy-only 平均为 <b>-0.147</b> 点（5/2/8）。CoLA/RTE 固定 seed42 的五子集均值分别为 -0.298、-0.361。跨子集 rank 距离三任务均增大，不能声称共识普遍提高分配稳定性。')
p('（2）MRPC 12 个预算守恒的局部 rank 交换中，共识效用对最终 Accuracy 变化的 Spearman 相关为 <b>-0.557</b>。几何重现相关不等于真实局部训练收益预测。')
p('（3）较大预算下 CoLA/MRPC 的自适应对均匀平均差为负；默认预算结果不能外推所有预算。三种子配对均值仅为描述性统计，QNLI/QQP 的 λ 曾使用验证集筛选。')
p('（4）论文式(8)的归一化分母与冻结实现不完全一致：代码除前 r_max 个共识修正谱项，还计入未修正的聚合梯度谱尾部。投稿前应按实现修正文稿。')

h('5. 论文落点与来源')
p('正文建议补一张错配共识表、一张评分方向表，并在 Discussion 同时呈现跨校准与局部效用负结果。公平的 GoRA/DPLoRA 同协议外部基线仍是后续优先事项。', 'CSmall')
p('完整逐种子与英文稿：/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1/analysis/GPU3_消融结果_20261107.md；GPU3_SHUFFLE_EXTENSION.csv；GPU3_SCORE_SIDE.csv；SHUFFLE_CONSENSUS.csv；PAPER_INSERT_20261107.md；PAPER_TABLES_20261107.tex。', 'CSmall')
p('原始组件与负向结果：experiments/extended_ablation_20261021_v2/analysis/REPORT.md；experiments/mrpc_calibration_robustness_v1/REPORT.md；experiments/rank_utility_predictivity_v1/REPORT.md。运行审计：experiments/speccon_targeted_ablation_v1/RUN_AUDIT.md。', 'CSmall')
p('本轮新增方向与错配训练共 33 次（首批 6 次，后续 GPU 3 为 9+12+6 次），全部留有正式结果；不计原 219 次扩展消融及复用参考。', 'CSmall')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor('#617282'))
    canvas.drawString(17*mm, 11*mm, 'SpecCon-LoRA 消融实验阶段报告 · 2026-11-07')
    canvas.drawRightString(193*mm, 11*mm, str(doc.page))
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=(210*mm, 297*mm), leftMargin=17*mm,
                  rightMargin=17*mm, topMargin=16*mm, bottomMargin=18*mm,
                  title='SpecCon-LoRA 消融实验阶段报告 2026-11-07').build(
    story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
