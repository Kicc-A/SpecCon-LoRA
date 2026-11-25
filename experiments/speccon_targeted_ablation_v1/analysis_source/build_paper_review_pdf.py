"""Append a clearly labeled review insert to the supplied PDF manuscript."""
from pathlib import Path
import fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path('/SpecCon-LoRA.pdf')
ANALYSIS = Path('/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1/analysis')
INSERT = ANALYSIS / 'PAPER_INSERT_20261107.pdf'
REVIEW = Path('/SpecCon-LoRA_消融增补审阅版_20261107.pdf')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX', parent=styles['Title'], fontName='Helvetica-Bold',
                          fontSize=16, leading=21, alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle(name='BodyX', parent=styles['BodyText'], fontName='Helvetica',
                          fontSize=8.9, leading=13, spaceAfter=7))
styles.add(ParagraphStyle(name='HeadX', parent=styles['Heading2'], fontName='Helvetica-Bold',
                          fontSize=11, leading=16, spaceBefore=11, spaceAfter=5))
styles.add(ParagraphStyle(name='CellX', parent=styles['BodyText'], fontName='Helvetica',
                          fontSize=7.8, leading=11))
story = []


def paragraph(text, style='BodyX'):
    story.append(Paragraph(text, styles[style]))


def table(headers, rows, widths):
    data = [[Paragraph(str(v), styles['CellX']) for v in headers]]
    data += [[Paragraph(str(v), styles['CellX']) for v in row] for row in rows]
    item = Table(data, colWidths=widths, repeatRows=1)
    item.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#d7e0e8')),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(item)
    story.append(Spacer(1, 7))


paragraph('Proposed Consensus-Ablation Insert', 'TitleX')
paragraph('Review supplement to the supplied 12-page SpecCon-LoRA PDF. This material is proposed manuscript text and tables; the original manuscript pages are unchanged.', 'BodyX')
paragraph('1. Shuffled-consensus control', 'HeadX')
paragraph('We deranged complete prefix-consensus curves among modules of identical matrix shape and type, retaining each receiving module\'s spectral energy. The original allocator, gradient-SVD initialization, 755,712-parameter budget, classifier initialization, training RNG, and final-checkpoint selection were held fixed. Both shuffle seeds were fixed before training. The original correspondence outperformed the two shuffles on CoLA and MRPC in mean performance, and the first shuffle on RTE. The MRPC advantage decreased substantially under the second shuffle, so both controls are shown.', 'BodyX')
table(['Task', 'Shuffle', 'Full', 'Shuffled', 'Full - Shuffled', 'W/T/L'], [
    ['CoLA', '1', '63.978 +/- 0.575', '61.436 +/- 0.793', '+2.543 +/- 1.287', '3/0/0'],
    ['CoLA', '2', '63.978 +/- 0.575', '62.329 +/- 1.982', '+1.649 +/- 2.072', '2/1/0'],
    ['MRPC', '1', '89.788 +/- 0.928', '87.663 +/- 1.234', '+2.124 +/- 0.861', '3/0/0'],
    ['MRPC', '2', '89.788 +/- 0.928', '89.461 +/- 0.884', '+0.327 +/- 0.566', '1/2/0'],
    ['RTE', '1', '80.505 +/- 0.625', '79.061 +/- 1.251', '+1.444 +/- 1.083', '3/0/0'],
], [20*mm, 17*mm, 36*mm, 36*mm, 45*mm, 20*mm])
paragraph('Scores are mean +/- sample SD over seeds 42, 777 and 2026. Paired differences use the same seeds. CoLA is MCC x 100; MRPC/RTE are Accuracy x 100. The two shuffles reuse the same training seeds and are not independent six-seed replications.', 'BodyX')
paragraph('2. Left, right, and joint consensus', 'HeadX')
paragraph('We compared the proposed leading left-singular-subspace overlap with right-subspace overlap and their geometric mean at each rank prefix. The same randomized SVD sketch and singular energies, score floor, consensus exponent, budget, initialization, and training recipes were used. Left-side consensus exceeded right-side consensus on all three paired seeds in CoLA, MRPC, and RTE. Its advantage over joint scoring was smaller and not unanimous.', 'BodyX')
table(['Task', 'Left', 'Right', 'Joint', 'Left - Right', 'Left - Joint'], [
    ['CoLA', '63.978 +/- 0.575', '61.793 +/- 0.845', '63.666 +/- 0.398', '+2.185 +/- 1.051', '+0.313 +/- 0.645'],
    ['MRPC', '89.788 +/- 0.928', '88.317 +/- 1.105', '88.889 +/- 0.374', '+1.471 +/- 0.490', '+0.899 +/- 1.020'],
    ['RTE', '80.505 +/- 0.625', '78.700 +/- 0.000', '79.061 +/- 1.910', '+1.805 +/- 0.625', '+1.444 +/- 2.527'],
], [18*mm, 30*mm, 30*mm, 30*mm, 34*mm, 34*mm])
paragraph('Left minus Right: 3/0/0 paired wins/ties/losses for each task. Left minus Joint: 2/0/1 for each task. Joint score = sqrt(C_left x C_right) before floor clamping and exponentiation.', 'BodyX')
paragraph('3. Discussion and method correction', 'HeadX')
paragraph('These controls support consensus correspondence and the output-side choice under the reference calibration. They do not establish calibration robustness. Across five alternative MRPC calibration draws and three training seeds, consensus minus energy averages -0.147 points (5 wins, 2 ties, 8 losses), and consensus increases cross-draw rank-map distance. An exploratory 12-swap MRPC study also finds a negative Spearman correlation (-0.557) between consensus utility and observed local trained rank utility. These are distinct from the positive held-out geometric recurrence correlations and should be reported together.', 'BodyX')
paragraph('Before submission, revise manuscript Eq. (8): the frozen allocator normalizes by the sum of consensus-adjusted candidate energies through r_max plus the unadjusted aggregated-gradient squared-singular-value tail, rather than only the first r_max adjusted terms. The editable tables and full draft text are in PAPER_TABLES_20261107.tex and PAPER_INSERT_20261107.md.', 'BodyX')

SimpleDocTemplate(str(INSERT), pagesize=(210*mm, 297*mm), leftMargin=17*mm,
                  rightMargin=17*mm, topMargin=14*mm, bottomMargin=14*mm,
                  title='Proposed SpecCon-LoRA consensus ablation insert').build(story)

base = fitz.open(BASE)
extra = fitz.open(INSERT)
base.insert_pdf(extra)
base.set_metadata({**base.metadata, 'title': 'SpecCon-LoRA with proposed consensus-ablation review insert'})
REVIEW.unlink(missing_ok=True)
base.save(REVIEW)
print(REVIEW, 'pages', len(base), 'insert_pages', len(extra))
