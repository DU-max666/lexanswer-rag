"""法衡 · 文档加载器。

统一入口 ``load_document(path)`` 按扩展名路由到具体加载器：

- ``.txt`` / ``.md``  纯文本（示例法规、咨询记录）
- ``.pdf``            文本型 PDF（pypdf）
- ``.png`` / ``.jpg`` 扫描件（OCR 接口层，未装 OCR 引擎时降级为占位提示）
- ``.docx``           Word（python-docx）

所有第三方依赖均为**可选**：缺失时抛出带安装提示的 ``LoaderUnavailable``，
不影响纯文本链路在裸环境跑通。

文本头部约定（用于解析元数据，见 sample_data 示例）::

    # 法名: 示例数据保护法
    # 效力: effective | 版本: 2024-06 | 敏感级: public | 机关: 示例立法机关
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

from .schema import DocMeta, EffectStatus, RawDoc, Sensitivity

_KV_RE = re.compile(r"^\s*(法名|效力|版本|敏感级|机关)\s*[:：]\s*(.+?)\s*$")


class LoaderUnavailable(RuntimeError):
    """依赖缺失时抛出，消息内含 pip 安装建议。"""


# ---------------------------------------------------------------------------
# 元数据头解析
# ---------------------------------------------------------------------------

def parse_meta_header(lines: List[str], path: str) -> Dict[str, object]:
    """解析文本头部的 ``# 键: 值`` 元数据行（支持 ``|`` 分隔的单行多字段）。"""
    fields: Dict[str, str] = {}
    idx = 0
    while idx < len(lines) and idx < 10:
        line = lines[idx].strip()
        if line == "":
            idx += 1
            continue
        if not line.startswith("#"):
            break
        for seg in line.lstrip("#").split("|"):
            m = _KV_RE.match(seg)
            if m:
                fields[m.group(1)] = m.group(2)
        idx += 1
    law_name = fields.get("法名", os.path.splitext(os.path.basename(path))[0])
    effect = fields.get("效力", "effective")
    sens = fields.get("敏感级", "public")
    meta = DocMeta(
        doc_id=DocMeta.normalize_law_name(law_name),
        law_name=DocMeta.normalize_law_name(law_name),
        source_path=path,
        effect_status=EffectStatus(effect),
        sensitivity=Sensitivity(sens),
        version=fields.get("版本", ""),
        issuer=fields.get("机关", ""),
    )
    return {"meta": meta, "body_lines": lines[idx:]}


# ---------------------------------------------------------------------------
# 各加载器
# ---------------------------------------------------------------------------

def load_text(path: str) -> RawDoc:
    """加载纯文本法规/记录。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    parsed = parse_meta_header(lines, path)
    meta: DocMeta = parsed["meta"]
    body = "\n".join(parsed["body_lines"])
    tables = re.findall(r"【附表(.*?)【完】", body, flags=re.S)
    return RawDoc(meta=meta, text=body, tables=["【附表" + t for t in tables])


def load_pdf(path: str) -> RawDoc:
    """加载文本型 PDF；未安装 pypdf 时给出明确提示。"""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - 依赖可选
        raise LoaderUnavailable("解析 PDF 需要 `pip install pypdf`") from exc
    reader = PdfReader(path)
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        pages.append(f"[PAGE:{i + 1}]\n" + (page.extract_text() or ""))
    parsed = parse_meta_header("\n".join(pages).splitlines(), path)
    return RawDoc(meta=parsed["meta"], text="\n".join(parsed["body_lines"]))


def load_scanned(path: str) -> RawDoc:
    """扫描件 OCR 接口层。

    生产环境接 PaddleOCR / RapidOCR；此处保持接口稳定，
    未安装引擎时返回带占位标记的文本，避免整条流水线崩溃。
    """
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        engine = RapidOCR()
        result, _ = engine(path)
        text = "\n".join(line[1] for line in (result or []))
    except ImportError:
        text = f"[OCR占位] 扫描件 {os.path.basename(path)} 未接入 OCR 引擎，请安装 rapidocr-onnxruntime 后重新入库。"
    parsed = parse_meta_header(text.splitlines(), path)
    meta: DocMeta = parsed["meta"]
    meta.extra["needs_ocr"] = True
    return RawDoc(meta=meta, text="\n".join(parsed["body_lines"]))


def load_docx(path: str) -> RawDoc:
    """加载 Word 文档，段落 + 表格分别抽取。"""
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:  # pragma: no cover - 依赖可选
        raise LoaderUnavailable("解析 Word 需要 `pip install python-docx`") from exc
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    tables: List[str] = []
    for tb in doc.tables:
        rows = ["\t".join(c.text.strip() for c in r.cells) for r in tb.rows]
        tables.append("【附表 表格】\n" + "\n".join(rows) + "\n【完】")
    parsed = parse_meta_header(paras, path)
    return RawDoc(meta=parsed["meta"], text="\n".join(parsed["body_lines"]), tables=tables)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

_LOADERS = {
    ".txt": load_text,
    ".md": load_text,
    ".pdf": load_pdf,
    ".png": load_scanned,
    ".jpg": load_scanned,
    ".jpeg": load_scanned,
    ".docx": load_docx,
}


def load_document(path: str) -> RawDoc:
    """按扩展名路由加载单个文档。"""
    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"不支持的文档类型: {ext}（支持 {sorted(_LOADERS)}）")
    return loader(path)


def load_directory(dir_path: str) -> List[RawDoc]:
    """批量加载目录下所有受支持文档。"""
    docs: List[RawDoc] = []
    for name in sorted(os.listdir(dir_path)):
        path = os.path.join(dir_path, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in _LOADERS:
            docs.append(load_document(path))
    return docs
