"""Render the dated Chinese ablation review with embedded CJK fonts."""
from pathlib import Path
import json
import hashlib
import re
import tempfile
import fitz
import markdown
import pypdf
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
STEM = 'SpecCon-LoRA_消融实验总结与论文佐证_20261025'
source = ROOT / (STEM + '.md')
target = ROOT / (STEM + '.pdf')
parts = source.read_text().split('<!-- PAGE -->')
css = '''
@font-face {font-family: cjk; src:url("NotoSansCJK-Regular.ttc");}
@font-face {font-family: cjk; src:url("NotoSansCJK-Bold.ttc");font-weight:bold;}
body {font-family:cjk; font-size:10pt; line-height:1.42; color:#243348;}
h1 {font-size:20pt; color:#142e49; margin:0 0 12pt;}
h2 {font-size:14pt; color:#142e49; margin:8pt 0 9pt;}
h3 {font-size:11pt; color:#206978; margin:10pt 0 5pt;}
p {margin:6pt 0;}
table {border-collapse:collapse; width:100%; margin:9pt 0; font-size:9pt;}
th {background-color:#e5eef4; color:#16374c; font-weight:bold;}
th, td {border:0.5pt solid #c9d6e2; padding:5pt; vertical-align:top;}
li {margin:5pt 0;}
blockquote {background-color:#eef4f7; border-left:3pt solid #397083; padding:10pt; margin:12pt 0;}
code {font-family:cjk; font-size:8pt;}
'''
pages = [markdown.markdown(p, extensions=['tables']) for p in parts]
# Break long source paths at slashes without changing the readable content.
pages[-1] = re.sub(r'(?<!<)/', '/\u200b', pages[-1])
(ROOT / (STEM + '.html')).write_text('<html><meta charset="utf-8"><style>' + css + '</style><body>' + '<div style="page-break-before:always"></div>'.join(pages) + '</body></html>')
with tempfile.TemporaryDirectory(prefix='ablation_review_') as temp:
    raw = Path(temp) / 'raw.pdf'
    writer = fitz.DocumentWriter(str(raw))
    media = fitz.paper_rect('a4')
    for index, body in enumerate(pages):
        page_css = css
        if index == 3:
            page_css += 'body {font-size:9pt;line-height:1.3;} h2 {font-size:13pt;margin:7pt 0;} p {margin:5pt 0;} li {margin:3pt 0;} table {font-size:7.8pt;} th,td {padding:3pt;} blockquote {margin:8pt 0;padding:8pt;}'
        story = fitz.Story(html='<body>' + body + '</body>', user_css=page_css,
                           archive=fitz.Archive('/usr/share/fonts/opentype/noto'))
        area = fitz.Rect(40, 43, media.width-40, media.height-43)
        more, filled = story.place(area)
        if more:
            raise RuntimeError(f'Page {index+1} overflows; adjust layout before publishing.')
        dev = writer.begin_page(media)
        story.draw(dev)
        writer.end_page()
    writer.close()
    with fitz.open(raw) as doc:
        for i, page in enumerate(doc):
            page.draw_line((40, 809), (media.width-40, 809), color=(.78,.83,.88), width=.5)
            page.insert_text((40, 824), 'SpecCon-LoRA | Ablation evidence review | 2026-10-25', fontsize=8, color=(.35,.43,.50))
            page.insert_text((media.width-58, 824), f'{i+1} / {len(doc)}', fontsize=8, color=(.35,.43,.50))
        doc.set_metadata({'title':'SpecCon-LoRA 消融实验总结与论文佐证', 'subject':'Completed ablations, supporting evidence and limitations', 'author':'SpecCon-LoRA experiment review'})
        doc.set_toc([[1, label, i+1] for i,label in enumerate(['核心证据','敏感性与预算','负向结果与边界','论文表述与来源'])])
        doc.subset_fonts(fallback=True)
        doc.save(target, garbage=4, clean=True, deflate=True, deflate_fonts=True, use_objstms=0)

assert len(pypdf.PdfReader(target, strict=True).pages) == 4
with fitz.open(target) as doc:
    text='\n'.join(page.get_text() for page in doc)
    for expected in ['同预算','219/219','0.298','1.471','跨子集','论文']:
        assert expected in text, expected
    assert '\ufffd' not in text
    text_counts = [len(page.get_text()) for page in doc]
    assert min(text_counts)>300
    fonts = [len(page.get_fonts()) for page in doc]
    previews=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.1,1.1))
        path=ROOT / f'{STEM}_page{i+1}.png'
        pix.save(path)
        im=Image.open(path).convert('RGB')
        im.thumbnail((490,700))
        previews.append(im)
    contact=Image.new('RGB',(1020,1450),'#e5e9ee')
    for i,im in enumerate(previews): contact.paste(im,(15+(i%2)*505,15+(i//2)*720))
    contact.save(ROOT / (STEM + '_preview.png'))
with pdfium.PdfDocument(str(target)) as independent:
    assert len(independent)==4
    for page in independent:
        bitmap=page.render(scale=.5)
        assert bitmap.width>0 and bitmap.height>0
        bitmap.close()
        page.close()
validation={'pdf':str(target),'pages':4,'bytes':target.stat().st_size,
    'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'text_characters_per_page':text_counts,'font_entries_per_page':fonts,
    'strict_parse':True,'cjk_text_extraction':True,'pdfium_all_pages_rendered':True}
(ROOT / (STEM + '_validation.json')).write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(validation,ensure_ascii=False,indent=2))
