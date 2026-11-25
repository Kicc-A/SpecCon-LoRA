#!/usr/bin/env python3
"""Rebuild and validate the existing result PDFs without changing experiment data."""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import fitz
import pypdf
import pypdfium2 as pdfium


ROOT = Path("/workspace/SpecCon-LoRA")
PACKAGE = ROOT / "reports/all_ablation_summary_20261020"
DELIVERY = ROOT / "reports/ablation_download_v2"
OVERVIEW = Path("00_总览/全部消融总总结.pdf")
COMBINED = Path("00_总览/全部消融_总览与完整表图.pdf")
TABLES = Path("07_论文表格与PDF/ACL表格源码/acl_tables.pdf")
MECHANISM = Path("05_独立校准机制/mechanism_report.pdf")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(message):
    print(message, flush=True)


def rewrite(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source) as original, fitz.open() as rebuilt:
        rebuilt.insert_pdf(original)
        rebuilt.set_metadata(original.metadata)
        rebuilt.set_toc(original.get_toc())
        rebuilt.save(
            target, garbage=4, clean=True, deflate=True,
            deflate_fonts=True, use_objstms=0,
        )


def validate(source, rebuilt):
    """Check text, exact MuPDF raster output, and independent PDFium rendering."""
    with fitz.open(source) as old, fitz.open(rebuilt) as new:
        assert not new.is_repaired and not new.needs_pass, rebuilt
        assert len(old) == len(new), rebuilt
        assert len(pypdf.PdfReader(rebuilt, strict=True).pages) == len(new), rebuilt
        with pdfium.PdfDocument(str(rebuilt)) as independent:
            assert len(independent) == len(new), rebuilt
            for index, (old_page, new_page) in enumerate(zip(old, new)):
                assert old_page.get_text() == new_page.get_text(), (rebuilt, index, "text")
                old_pix = old_page.get_pixmap()
                new_pix = new_page.get_pixmap()
                assert old_pix.samples == new_pix.samples, (rebuilt, index, "pixels")
                page = independent[index]
                bitmap = page.render(scale=1)
                assert bitmap.width > 0 and bitmap.height > 0, (rebuilt, index)
                bitmap.close()
                page.close()
        return {
            "path": str(source.relative_to(PACKAGE)),
            "pages": len(new),
            "old_bytes": source.stat().st_size,
            "new_bytes": rebuilt.stat().st_size,
            "old_sha256": sha(source),
            "new_sha256": sha(rebuilt),
            "text_unchanged": True,
            "mupdf_pixels_unchanged": True,
            "pdfium_all_pages_rendered": True,
            "pypdf_strict_parse": True,
        }


def create_compatibility_pdf(source, target):
    """A font-independent PDF for readers that struggle with embedded CJK fonts."""
    with pdfium.PdfDocument(str(source)) as original, fitz.open() as output:
        for index in range(len(original)):
            page = original[index]
            width, height = page.get_size()
            bitmap = page.render(scale=2.5)
            image = bitmap.to_pil()
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92, subsampling=0)
            output.new_page(width=width, height=height).insert_image(
                fitz.Rect(0, 0, width, height), stream=buffer.getvalue()
            )
            image.close()
            bitmap.close()
            page.close()
        output.set_metadata({
            "title": "SpecCon-LoRA All Ablations - Compatibility Edition",
            "subject": "Image-based PDF, 180 dpi; use all_ablations.pdf for selectable text.",
        })
        output.save(target, garbage=4, deflate=True, use_objstms=0)
    with pdfium.PdfDocument(str(target)) as check:
        assert len(check) == len(pypdf.PdfReader(target, strict=True).pages) == 19
        for index in range(len(check)):
            page = check[index]
            bitmap = page.render(scale=0.7)
            assert bitmap.width and bitmap.height
            bitmap.close()
            page.close()


def main():
    DELIVERY.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ablation_pdf_rebuild_") as temp:
        stage = Path(temp)
        overview = stage / OVERVIEW
        overview.parent.mkdir(parents=True, exist_ok=True)
        log("Subsetting the large overview fonts with fontTools...")
        # Reuse the validated probe in this repair session, otherwise rebuild it.
        probe = Path("/tmp/ablation_summary_fonttools_probe.pdf")
        if probe.exists():
            shutil.copy2(probe, overview)
        else:
            with fitz.open(PACKAGE / OVERVIEW) as document:
                document.subset_fonts(fallback=True)
                document.save(
                    overview, garbage=4, clean=True, deflate=True,
                    deflate_fonts=True, use_objstms=0,
                )
        log("Rewriting the remaining PDF files...")
        pdfs = sorted(PACKAGE.rglob("*.pdf"))
        for source in pdfs:
            relative = source.relative_to(PACKAGE)
            if relative not in (OVERVIEW, COMBINED):
                rewrite(source, stage / relative)
        with fitz.open() as merged:
            toc = []
            for title, relative in [
                ("中文总总结", OVERVIEW),
                ("18张论文结果与消融表", TABLES),
                ("独立校准机制图", MECHANISM),
            ]:
                toc.append([1, title, len(merged) + 1])
                with fitz.open(stage / relative) as part:
                    merged.insert_pdf(part)
            merged.set_toc(toc)
            with fitz.open(PACKAGE / COMBINED) as old:
                merged.set_metadata(old.metadata)
            merged.save(
                stage / COMBINED, garbage=4, clean=True, deflate=True,
                deflate_fonts=True, use_objstms=0,
            )

        records = []
        for source in pdfs:
            relative = source.relative_to(PACKAGE)
            records.append(validate(source, stage / relative))
            log(f"Validated {len(records)}/{len(pdfs)}: {relative}")

        log("Creating the font-independent compatibility edition...")
        compatibility = stage / "all_ablations_compatible.pdf"
        create_compatibility_pdf(stage / COMBINED, compatibility)
        # Only replace old PDFs once every replacement has passed validation.
        for source in pdfs:
            relative = source.relative_to(PACKAGE)
            temporary = source.with_suffix(".replacement.pdf")
            shutil.copy2(stage / relative, temporary)
            os.replace(temporary, source)

        for relative, name in [
            (OVERVIEW, "summary.pdf"),
            (COMBINED, "all_ablations.pdf"),
            (TABLES, "tables.pdf"),
            (MECHANISM, "mechanism.pdf"),
        ]:
            shutil.copy2(PACKAGE / relative, DELIVERY / name)
        shutil.copy2(compatibility, DELIVERY / compatibility.name)

    # Keep source provenance explicit after changing only PDF serialization.
    mapping = PACKAGE / "08_复现与溯源/复制文件来源映射.csv"
    with mapping.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    extras = ["original_bytes_before_pdf_rebuild", "original_sha256_before_pdf_rebuild",
              "pdf_processing"]
    fields = fields + [field for field in extras if field not in fields]
    changed = {row["path"]: row for row in records}
    for row in rows:
        if row["archive_path"] in changed:
            record = changed[row["archive_path"]]
            row["original_bytes_before_pdf_rebuild"] = row["bytes"]
            row["original_sha256_before_pdf_rebuild"] = row["sha256"]
            row["bytes"] = record["new_bytes"]
            row["sha256"] = record["new_sha256"]
            row["pdf_processing"] = "PDF rebuilt; text and rendered pixels unchanged"
    with mapping.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "pdf_count": len(records),
        "page_count": sum(row["pages"] for row in records),
        "checks": ["unchanged text", "unchanged MuPDF page pixels",
                   "PDFium every-page rendering", "strict pypdf parsing"],
        "compatibility_edition_pages": 19,
        "pdfs": records,
    }
    (PACKAGE / "PDF_REBUILD_VALIDATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    validation_path = PACKAGE / "VALIDATION.json"
    validation = json.loads(validation_path.read_text())
    validation["pdf_rebuild"] = {
        key: value for key, value in report.items() if key != "pdfs"
    }
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(Path(__file__), PACKAGE / "08_复现与溯源/启动与汇总脚本" / Path(__file__).name)
    checksum_path = PACKAGE / "SHA256SUMS.txt"
    files = sorted(p for p in PACKAGE.rglob("*") if p.is_file() and p != checksum_path)
    checksum_path.write_text("".join(f"{sha(p)}  {p.relative_to(PACKAGE).as_posix()}\n" for p in files))

    log("Repacking the complete result archive...")
    archive = PACKAGE.with_suffix(".zip")
    temporary = archive.with_suffix(".replacement.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file()):
            zipped.write(path, PACKAGE.name + "/" + path.relative_to(PACKAGE).as_posix())
    with zipfile.ZipFile(temporary) as zipped:
        assert zipped.testzip() is None
        assert sum(name.endswith(".pdf") for name in zipped.namelist()) == len(records)
    os.replace(temporary, archive)
    archive.with_suffix(".zip.sha256").write_text(f"{sha(archive)}  {archive.name}\n")
    shutil.copy2(archive, DELIVERY / "all_ablations_package_v2.zip")
    shutil.copy2(
        PACKAGE / "00_总览/全部消融_可编辑数据.xlsx", DELIVERY / "ablation_data.xlsx"
    )
    (DELIVERY / "README.txt").write_text(
        "SpecCon-LoRA 消融结果重新下载版\n\n"
        "summary.pdf：3 页中文总结，可选择和复制文字。\n"
        "all_ablations.pdf：19 页完整总结与表图，可选择和复制文字。\n"
        "all_ablations_compatible.pdf：19 页图片兼容版，无中文字体依赖，文字不可选择。\n"
        "all_ablations_package_v2.zip：重新替换全部 39 个 PDF 的完整资料包。\n"
        "ablation_data.xlsx：可编辑数据。\n\n"
        "39 个 PDF 共 102 页已通过双引擎渲染、文本/画面一致性和严格结构解析检查。\n"
        "重新导出的结果内容与原版一致。\n",
        encoding="utf-8",
    )
    (DELIVERY / "SHA256SUMS.txt").write_text(
        "".join(f"{sha(p)}  {p.name}\n" for p in sorted(DELIVERY.iterdir())
                if p.is_file() and p.name != "SHA256SUMS.txt")
    )
    log(json.dumps({
        "delivery": str(DELIVERY),
        "replaced_pdfs": len(records),
        "validated_pages": report["page_count"],
        "files": {p.name: p.stat().st_size for p in sorted(DELIVERY.iterdir())},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
