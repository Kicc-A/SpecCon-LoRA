#!/usr/bin/env python3
import csv
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

import fitz
import markdown
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from package_all_ablations_20261020 import ROOT,DEST,DIRS,MECH,MEXP,copy,sha,csvout,dump,log,PROVENANCE

PAPER=DEST/DIRS[7]/'ACL表格源码'
SNAP=json.loads((PAPER/'snapshot.json').read_text())


def copy_reports_and_sources():
    log('Copying categorized reports and raw metric evidence...')
    for p in MECH.iterdir():
        if p.is_file() and not p.name.startswith('preview_'):
            copy(p,DEST/DIRS[5]/p.name)
    for name in ['PROTOCOL.md','REPRODUCE.md','manifest.json']:
        copy(MEXP/name,DEST/DIRS[5]/'实验协议'/name)
    # Small per-run records include all ranks, 100 permutations and spectral profiles as JSON.
    for p in (MEXP/'runs').rglob('*.json'):
        copy(p,DEST/DIRS[5]/'逐校准记录'/p.relative_to(MEXP/'runs'))
    for p in (MEXP/'independent_spectral_seed_check').iterdir():
        if p.suffix in ['.json','.py']:copy(p,DEST/DIRS[5]/'独立谱种子复核'/p.name)
    for p in (MEXP/'implementation_history').glob('*'):
        if p.is_file():copy(p,DEST/DIRS[8]/'机制实验数值修复记录'/p.name)
    tensor_index=[]
    for taskdir in (MEXP/'runs').iterdir():
        done=json.loads((taskdir/'DONE.json').read_text())
        for p in taskdir.glob('*.pt'):
            tensor_index.append(dict(path=str(p),bytes=p.stat().st_size,sha256_recorded=done['artifacts'][p.name],
                                     included_in_zip=False,kind='calibration gradient / basis cache'))
    csvout(DEST/DIRS[8]/'大型原始缓存位置索引.csv',tensor_index)
    table_categories={1:[2,3,12,16],2:[5,6],3:[4,14],4:[15],5:[17,18],6:[1,7,8,9,10,13]}
    tables=json.loads((PAPER/'table_data.json').read_text())
    for category,ids in table_categories.items():
        for index in ids:
            t=tables[index-1]
            copy(PAPER/'csv'/(t['id']+'.csv'),DEST/DIRS[category]/(t['id']+'.csv'))
    formal={'glue_main_ablation_3seeds_gpu12_20261016_v3','glue_rdrop_ablation_seed42_gpu3_20261016_v2',
        'glue_paper_extras_gpu3_20261016_v1','glue_lambda0_seed42_gpu2_20261015_v1',
        'rte_qnli_lambda15_3seeds_gpu012_20261015_v1','qnli_lambda07_3seeds_gpu0_20261016_v1'}
    excluded={'glue_main_ablation_3seeds_gpu12_20261016_v1','glue_main_ablation_3seeds_gpu12_20261016_v2',
              'glue_rdrop_ablation_seed42_gpu3_20261016_v1'}
    inventory=[];historical_metrics=[]
    for exp in sorted((ROOT/'experiments').iterdir()):
        if not exp.is_dir():continue
        name=exp.name
        kind='正式消融' if name in formal else ('失败预检版本' if name in excluded else
             ('独立机制' if name==MEXP.name else ('背景任务' if name.startswith(('llama31','mtbench')) else '历史探索或保留主结果')))
        root_files=[p for p in exp.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
        for p in root_files:copy(p,DEST/DIRS[8]/'实验清单与状态'/name/p.name)
        if kind not in ['失败预检版本','背景任务']:
            for p in (exp/'source_snapshot').rglob('*'):
                if p.is_file() and p.suffix in ['.py','.sh','.json','.md','.yaml','.toml'] and '__pycache__' not in p.parts:
                    copy(p,DEST/DIRS[8]/'冻结源码'/name/p.relative_to(exp/'source_snapshot'))
        complete=[];other=[]
        for p in (exp/'runs').rglob('metrics.json'):
            try:d=json.loads(p.read_text())
            except (ValueError,UnicodeDecodeError):continue
            if d.get('state')=='complete' and d.get('epochs')==d.get('planned_epochs') and not d.get('stopped_early'):
                complete.append(p)
                if kind not in ['失败预检版本','独立机制','背景任务']:
                    for f in p.parent.iterdir():
                        if f.is_file() and (f.suffix in ['.json','.jsonl'] or f.name=='training_rng.pt'):
                            copy(f,DEST/DIRS[8]/'运行原始记录'/name/f.relative_to(exp))
                if kind=='历史探索或保留主结果':
                    t=d['task'];sp=d['final']['splits'];field={'cola':'matthews_correlation','stsb':'pearson'}.get(t,'accuracy')
                    score=50*(sp['matched']['accuracy']+sp['mismatched']['accuracy']) if t=='mnli' else 100*sp['validation'][field]
                    hp=d['hyperparameters']
                    historical_metrics.append(dict(campaign=name,task=t,seed=d['seed'],score=score,epochs=d['epochs'],
                        lambda_value=hp.get('lambda'),gamma=hp.get('gamma'),learning_rate=hp.get('lr'),dropout=hp.get('dropout'),
                        label_smoothing=hp.get('label_smoothing'),rdrop_alpha=hp.get('rdrop_alpha'),
                        calibration_steps=hp.get('calibration_steps'),source=str(p)))
            else:other.append(str(p))
        manifest_file=next((p for p in root_files if p.name=='manifest.json'),None) or next((p for p in root_files if p.name=='launch_manifest.json'),None)
        manifest=json.loads(manifest_file.read_text()) if manifest_file else {}
        inventory.append(dict(campaign=name,category=kind,manifest_jobs=len(manifest.get('jobs',[])),
            retained_complete_metrics_in_runs=len(complete),other_metrics_in_runs=len(other),
            manifest=str(manifest_file) if manifest_file else '',
            note='仅统计当前runs目录留存文件；不含references复用。manifest任务数与该计数不是直接完成率。'))
    csvout(DEST/DIRS[6]/'历史保留结果_逐种子.csv',historical_metrics)
    csvout(DEST/DIRS[8]/'全部实验目录盘点.csv',inventory)
    for src in ROOT.joinpath('reproduction').glob('*.py'):
        if any(s in src.name for s in ['ablation','paper_extras','lambda07','independent_calibration','independent_spectral','main_ablation_support']):
            copy(src,DEST/DIRS[8]/'启动与汇总脚本'/src.name)
    for name in ['current_glue_manifest.json']:
        copy(ROOT/'reproduction'/name,DEST/DIRS[8]/'原主表来源清单'/name)
    for name in ['PAPER_ABLATION_PLAN_20261016.md','PAPER_EXPERIMENT_STATUS_20261016.md']:
        copy(ROOT/'reports'/name,DEST/DIRS[8]/'历史计划_状态不代表当前'/name)
    for name in ['CURRENT_GLUE.md','BEST_NLG_RESULTS.md']:
        copy(ROOT/name,DEST/DIRS[6]/'历史背景文档'/name)
    for dirname in ['glue_best_42_777_2026_20261013','mtbench_seed42_speccon_vs_dplora_20261014','dplora_rank_heatmaps_20261016','cola_rd10_retained_20261005']:
        src=ROOT/'reports'/dirname
        for p in src.rglob('*'):
            if p.is_file() and p.suffix in ['.md','.json','.csv','.pdf','.png'] and p.stat().st_size<6*1024*1024:
                copy(p,DEST/DIRS[6]/'历史背景文档'/dirname/p.relative_to(src))
    # Retain portable metric copies for every source actually used by the fresh tables.
    for original,meta in SNAP['source_files'].items():
        p=PAPER/meta['snapshot']
        assert sha(p)==meta['sha256'],p
    (DEST/DIRS[6]/'README_如何解读历史记录.md').write_text('''# 历史探索与补充对照

本目录与01–05的正式机制/消融分开阅读。这里包含保留主表、DPLoRA本地与论文参考、数学/代码、MT-Bench历史对照，以及当前仍有原始指标的调参结果。

历史RTE清单涉及学习率、gamma与lambda；QNLI/SST-2清单涉及多个lambda；MRPC/RTE涉及dropout与标签平滑。部分运行指标已经不在当前工作区，只留配置清单或筛选出的结果。历史保留结果_逐种子.csv只收录实际读到的完整末轮metrics，不填补缺失候选，也不把它描述为完整公平扫描。

保留结果与正式消融可能复用同一次训练；不能按文件数叠加实验次数。旧CURRENT_GLUE、旧历史报告中的QNLI选中lambda或队列状态可能过时，当前结论以00目录与07目录新生成表为准。

DPLoRA论文参考、本地复现与SpecCon训练配方/种子不同；数学/代码是保留的最高单次结果；MT-Bench历史配方和裁判日期不同。这些材料提供背景，不能作为隔离算法因素的严格消融。

目录盘点与配置清单见08目录。旧文档中的绝对路径指向原工作区，包内副本路径可通过08目录的来源映射查找。
''')
    return inventory,tensor_index


def render_summary():
    log('Rendering Chinese overview and merging the complete PDF...')
    md=(DEST/DIRS[0]/'全部消融总总结.md').read_text()
    body=markdown.markdown(md,extensions=['tables'])
    css='''
    @font-face {font-family: cjk; src:url("NotoSansCJK-Regular.ttc");}
    @font-face {font-family: cjk; src:url("NotoSansCJK-Bold.ttc");font-weight:bold;}
    body {font-family:cjk; font-size:9.2pt; line-height:1.48; color:#18212c;}
    h1 {font-size:20pt;color:#174561;margin-bottom:12pt;}
    h2 {font-size:13pt;color:#174561;margin-top:15pt;margin-bottom:7pt;page-break-after:avoid;}
    p {margin-top:5pt;margin-bottom:7pt;}
    table {border-collapse:collapse;width:100%;margin-top:6pt;margin-bottom:10pt;font-size:8.0pt;}
    th {background:#e9f1f5;font-weight:bold;}
    th,td {padding:4pt;border-bottom:0.5pt solid #cbd4dc;text-align:left;}
    tr {page-break-inside:avoid;}
    blockquote {border-left:3pt solid #247ba0;padding-left:10pt;color:#25475b;}
    '''
    (DEST/DIRS[0]/'全部消融总总结.html').write_text('<html><meta charset="utf-8"><style>'+css.replace('@font-face {font-family: cjk; src:url("NotoSansCJK-Regular.ttc");}','').replace('@font-face {font-family: cjk; src:url("NotoSansCJK-Bold.ttc");font-weight:bold;}','')+'</style><body>'+body+'</body></html>')
    story=fitz.Story(html='<body>'+body+'</body>',user_css=css,archive=fitz.Archive('/usr/share/fonts/opentype/noto'))
    out=DEST/DIRS[0]/'全部消融总总结.pdf'
    writer=fitz.DocumentWriter(str(out))
    more=1
    while more:
        page=fitz.paper_rect('a4');dev=writer.begin_page(page)
        more,_=story.place(fitz.Rect(40,40,page.width-40,page.height-42))
        story.draw(dev);writer.end_page()
    writer.close()
    merged=fitz.open();toc=[]
    for title,path in [('中文总总结',out),('18张论文结果与消融表',PAPER/'acl_tables.pdf'),('独立校准机制图',DEST/DIRS[5]/'mechanism_report.pdf')]:
        toc.append([1,title,len(merged)+1]);doc=fitz.open(path);merged.insert_pdf(doc)
    merged.set_toc(toc)
    merged.set_metadata({'title':'SpecCon-LoRA 全部消融实验总结 2026-10-20','author':'SpecCon-LoRA experiment archive'})
    merged.save(DEST/DIRS[0]/'全部消融_总览与完整表图.pdf',garbage=4,deflate=True)
    # Unified numeric workbook, with every CSV retained separately in its category.
    wb=Workbook();wb.remove(wb.active)
    mapping=[('完成情况',DEST/DIRS[0]/'完成情况.csv'),('组件逐种子',DEST/DIRS[1]/'逐种子结果.csv'),
        ('组件配对差值',DEST/DIRS[1]/'配对差值_三种子.csv'),('Lambda逐种子',DEST/DIRS[2]/'lambda_全部可核验逐种子.csv'),
        ('RDrop逐种子',DEST/DIRS[3]/'RDrop_36格逐种子.csv'),('Rank置换逐种子',DEST/DIRS[4]/'rank置换_逐种子.csv'),
        ('机制捕获',DEST/DIRS[5]/'captures.csv'),('机制预测性',DEST/DIRS[5]/'predictivity.csv'),
        ('机制重复性',DEST/DIRS[5]/'rank_distances.csv'),('历史保留结果',DEST/DIRS[6]/'历史保留结果_逐种子.csv')]
    for label,path in mapping:
        ws=wb.create_sheet(label)
        with open(path,encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.reader(f)):
                converted=[]
                for cell in row:
                    if i and re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',cell):
                        cell=float(cell) if any(x in cell for x in '.eE') else int(cell)
                    converted.append(cell)
                ws.append(converted)
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='174561')
        for col in range(1,ws.max_column+1):
            ws.column_dimensions[get_column_letter(col)].width=min(60,max(12,max(len(str(ws.cell(i,col).value or '')) for i in range(1,min(ws.max_row,80)+1))+2))
    wb.save(DEST/DIRS[0]/'全部消融_可编辑数据.xlsx')
    return len(fitz.open(out)),len(merged)


def validate_and_zip(inventory,tensors,overview_pages,combined_pages):
    log('Checking numerical sources, archive integrity and final files...')
    snapshots=0
    for original,meta in SNAP['source_files'].items():
        assert sha(Path(original))==sha(PAPER/meta['snapshot'])==meta['sha256']
        snapshots+=1
    assert len(SNAP['components'])==54 and all(x['result'] for x in SNAP['components'])
    assert len(SNAP['extended_rdrop'])==36 and all(x['result'] for x in SNAP['extended_rdrop'])
    assert len(SNAP['rank_shuffle'])==6 and all(x['result'] for x in SNAP['rank_shuffle'])
    rte0=next(x for x in SNAP['sensitivity'] if x['task']=='rte' and x['lambda']==0)
    assert sorted(x['seed'] for x in rte0['results'])==[42,777,2026]
    for taskdir in (MEXP/'runs').iterdir():
        done=json.loads((taskdir/'DONE.json').read_text())
        for name,expected in done['artifacts'].items():
            if name.endswith('.json'):assert sha(taskdir/name)==expected
    problems=[];pdfs=[]
    for p in [DEST/DIRS[0]/'全部消融总总结.pdf',PAPER/'acl_tables.pdf',DEST/DIRS[0]/'全部消融_总览与完整表图.pdf']:
        doc=fitz.open(p);pdfs.append(dict(path=str(p.relative_to(DEST)),pages=len(doc)))
        for page in doc:
            assert page.get_text().strip(),(p,page.number)
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox']
                        if min(x0,y0)<-1 or x1>page.rect.width+1 or y1>page.rect.height+1:
                            problems.append(dict(file=p.name,page=page.number+1,text=span['text']))
    assert not problems,problems
    doc=fitz.open(DEST/DIRS[0]/'全部消融总总结.pdf')
    assert '组件' in ''.join(p.get_text() for p in doc),'Chinese text missing'
    previews=DEST/DIRS[0]/'预览';previews.mkdir(exist_ok=True)
    for i in [0,min(2,len(doc)-1)]:doc[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(str(previews/f'总总结_第{i+1}页.png'))
    csvout(DEST/DIRS[8]/'复制文件来源映射.csv',PROVENANCE)
    readme='''# SpecCon-LoRA 全部消融实验包（2026-10-20）

建议先看 **00_总览/全部消融总总结.pdf**；完整表格和机制图合并在 **00_总览/全部消融_总览与完整表图.pdf**，可编辑数值见 **00_总览/全部消融_可编辑数据.xlsx**。

## 目录

|目录|内容|
|---|---|
|00_总览|中文总总结、合并PDF、Excel、未舍入统计、完成情况|
|01_组件与初始化|54格组件结果、三种子配对差值、方案定义与结论|
|02_lambda敏感性|八任务可核验lambda结果；RTE λ=0已补全三种子|
|03_RDrop消融|完整36格，逐种子结果与配方解释|
|04_rank置换|MRPC/RTE三种子原始与置换结果|
|05_独立校准机制|36次校准/复核的报告、图表、逐候选CSV、逐校准JSON与协议|
|06_历史调参与补充对照|保留调参记录、GLUE主表及DPLoRA/NLG/MT-Bench背景|
|07_论文表格与PDF|18张最新ACL表、CSV、Excel、LaTeX、采集源码与冻结输入|
|08_复现与溯源|实验清单、训练配置、原始metrics/历史曲线、rank表、冻结代码、文件来源|
|09_证据缺口与排除项|缺失历史结果、失败预检版本、未完成证据与口径限制|

## 本次修正

旧表的RTE λ=0只列seed42；新表复用已完成组件D，补齐42/777/2026。新增三种子同预算组件差值表及机制结果表。QNLI主表继续使用非零λ=0.7，λ=0作为消融。旧背景文档仅供溯源，可能含过时状态，以00和07为准。

训练使用固定末轮、样本标准差ddof=1；MRPC统一为Accuracy。跨类别存在复用，完成格数不能相加当独立训练次数。A/B与C–F预算不同。独立机制校准不等于下游训练，反例与正向结果均收录。

## 归档范围与复现

这是结果与复现材料包，包含数值、PDF、图、源码、配置、rank表和小型随机状态。模型权重、数据集和全量梯度/基底张量保留原工作区；大型缓存位置及已记录哈希见08目录索引。使用代码重跑训练需要相应本地模型、数据和路径映射；本包不是独立模型镜像。只查看表图、CSV、JSON与Excel不需要原工作区。

07目录的build_tables.py保留原15张表生成逻辑；完整18张及中文总包的生成流程由08/启动与汇总脚本中的package_all_ablations_20261020.py和finalize_all_ablation_package_20261020.py记录。脚本含原工作区绝对路径，迁移后需调整ROOT/输入路径；原始训练命令见各manifest。

05目录中的机制analysis脚本在原工作区可重算；逐候选CSV与逐校准JSON便于独立统计。大型缓存索引的哈希来自已完成实验记录，缓存张量本身未装入此结果包。

SHA256SUMS.txt校验包内各文件，VALIDATION.json记录数量、源文件核对与PDF检查。
'''
    (DEST/'README_先看这里.md').write_text(readme)
    (DEST/'BUILD_STATE.json').write_text(json.dumps({'stage':'complete'},indent=2)+'\n')
    validation=dict(component_cells=54,rdrop_cells=36,rank_permutation_training_runs=6,mechanism_calibrations=36,
        lambda0_seed42_tasks=8,corrected_rte_lambda0_seeds=[42,777,2026],fresh_tables=18,
        source_files_verified=snapshots,source_hash_mismatches=[],pdfs=pdfs,pdf_text_outside_page=problems,
        experiment_directories_indexed=len(inventory),large_tensors_indexed=len(tensors),
        large_tensor_bytes_outside_zip=sum(x['bytes'] for x in tensors),
        scope='Results/configuration/source bundle. Excludes model checkpoints, datasets and full gradient tensors.')
    dump(DEST/'VALIDATION.json',validation)
    # Copy both packaging scripts after all definitions are present.
    for name in ['package_all_ablations_20261020.py','finalize_all_ablation_package_20261020.py']:
        shutil.copy2(ROOT/'reproduction'/name,DEST/DIRS[8]/'启动与汇总脚本'/name)
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    checksum=''.join(sha(p)+'  '+p.relative_to(DEST).as_posix()+'\n' for p in files)
    (DEST/'SHA256SUMS.txt').write_text(checksum)
    archive=DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,DEST.name+'/'+p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)+1
        assert not any('..' in Path(n).parts or n.startswith('/') for n in z.namelist())
    zipsha=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(zipsha+'  '+archive.name+'\n')
    log(json.dumps(dict(zip=str(archive),bytes=archive.stat().st_size,files=len(files)+1,
                        overview_pages=overview_pages,combined_pdf_pages=combined_pages,sha256=zipsha),ensure_ascii=False,indent=2))


if __name__=='__main__':
    inventory,tensors=copy_reports_and_sources()
    pages,combined=render_summary()
    validate_and_zip(inventory,tensors,pages,combined)
