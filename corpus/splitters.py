"""法衡 · 按文档类型路由的切分器族。

四种切分器（与 ``docs/chunking.md`` 一一对应）：

1. ``RegulationSplitter``  法规：章/节/条/款 父子块，条块带《法名》第N条引头
2. ``TableSplitter``       附表：按行切，每行携带表头语义，父块为整表
3. ``CaseSplitter``        类案：语义断崖切分（相邻段相似度骤降处断开）
4. ``ConsultSplitter``     咨询记录：按问答轮次成对切分

``split_document(raw_doc)`` 依据元数据 ``extra["doc_type"]`` 或正文特征自动路由。
"""
from __future__ import annotations

import re
from typing import List, Optional

from .schema import Chunk, ChunkKind, DocMeta, EffectStatus, RawDoc, Sensitivity

# 中文数字 -> 阿拉伯数字（条号解析用）
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "两": 2}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

_ARTICLE_RE = re.compile(r"^第([零一二三四五六七八九十百千两0-9]+)条")
_CLAUSE_RE = re.compile(r"^（([零一二三四五六七八九十]+|[0-9]+)）")
_CHAPTER_RE = re.compile(r"^第([零一二三四五六七八九十百千]+)章\s*(.*)$")
_SECTION_RE = re.compile(r"^第([零一二三四五六七八九十百千]+)节\s*(.*)$")
_TABLE_BLOCK_RE = re.compile(r"【附表(.*?)】(.*?)【完】", flags=re.S)


def cn_to_int(token: str) -> Optional[int]:
    """将「二十三」「第108条中的 108」等条号记法转为整数，失败返回 None。"""
    token = token.strip()
    if token.isdigit():
        return int(token)
    total, current = 0, 0
    for ch in token:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if unit == 10 and current == 0:
                current = 1  # 「十五」隐含一十
            total += current * unit
            current = 0
        else:
            return None
    return total + current


def heading_of(meta: DocMeta, article_no: Optional[int]) -> str:
    """生成统一引头：《法名》第N条。"""
    if article_no is None:
        return f"《{meta.law_name}》"
    return f"《{meta.law_name}》第{_int_to_cn(article_no)}条"


def _int_to_cn(num: int) -> str:
    """阿拉伯数字转中文条号（仅到千位，法规条号足够覆盖）。"""
    if num < 10:
        return "零一二三四五六七八九"[num]
    units = [(1000, "千"), (100, "百"), (10, "十")]
    out, rest = "", num
    for value, name in units:
        if rest >= value:
            d = rest // value
            if d > 1:
                out += _int_to_cn(d) + name
            else:
                out += name
            rest %= value
    if rest:
        out += _int_to_cn(rest)
    return out


def _base_chunk(meta: DocMeta, kind: ChunkKind, text: str,
                article_no: Optional[int] = None,
                clause_no: Optional[int] = None,
                parent_id: Optional[str] = None) -> Chunk:
    """构造带统一元数据的块，chunk_id 由 文档+类型+条+款 稳定生成。"""
    return Chunk(
        chunk_id=Chunk.make_id(meta.doc_id, kind, article_no, clause_no),
        doc_id=meta.doc_id,
        law_name=meta.law_name,
        kind=kind,
        text=text,
        heading=heading_of(meta, article_no),
        article_no=article_no,
        clause_no=clause_no,
        parent_id=parent_id,
        effect_status=meta.effect_status,
        sensitivity=meta.sensitivity,
    )


# ---------------------------------------------------------------------------
# 1. 法规切分器：章/节/条/款 父子块
# ---------------------------------------------------------------------------

class RegulationSplitter:
    """法规切分器：条为父块（整条注入），款/项为子块（精确命中用）。

    父块文本自带引头 ``《法名》第N条``，子块通过 ``parent_id`` 回溯，
    这样向量检索命中「第X条第Y款」时，仍能向模型注入整条上下文。
    """

    def split(self, raw: RawDoc) -> List[Chunk]:
        meta = raw.meta
        lines = [ln.strip() for ln in raw.text.splitlines()]
        chunks: List[Chunk] = []
        cur_article_lines: List[str] = []
        cur_article_no: Optional[int] = None
        cur_clause_lines: List[str] = []
        cur_clause_no: Optional[int] = None

        def flush_clause() -> None:
            nonlocal cur_clause_lines, cur_clause_no
            if cur_clause_lines and cur_article_no is not None:
                parent_id = Chunk.make_id(meta.doc_id, ChunkKind.ARTICLE, cur_article_no)
                c = _base_chunk(
                    meta, ChunkKind.CLAUSE,
                    "\n".join(cur_clause_lines),
                    article_no=cur_article_no, clause_no=cur_clause_no,
                    parent_id=parent_id,
                )
                chunks.append(c)
            cur_clause_lines, cur_clause_no = [], None

        def flush_article() -> None:
            nonlocal cur_article_lines, cur_article_no
            if cur_article_lines and cur_article_no is not None:
                # 首行的「第N条」与前缀引头重复，剥离后仅保留条文正文
                first = cur_article_lines[0]
                m = _ARTICLE_RE.match(first)
                if m:
                    first = first[m.end():].strip()
                body = "\n".join([first] + cur_article_lines[1:]).strip()
                parent = _base_chunk(
                    meta, ChunkKind.ARTICLE,
                    heading_of(meta, cur_article_no) + " " + body,
                    article_no=cur_article_no,
                )
                # 子块指针回填
                parent.child_ids = [
                    c.chunk_id for c in chunks
                    if c.parent_id == parent.chunk_id
                ]
                chunks.append(parent)
            cur_article_lines, cur_article_no = [], None

        for line in lines:
            if not line:
                continue
            art = _ARTICLE_RE.match(line)
            cla = _CLAUSE_RE.match(line)
            ch = _CHAPTER_RE.match(line) or _SECTION_RE.match(line)
            if ch:
                # 章/节标题：结束当前条，独立成低优先级上下文块
                flush_clause()
                flush_article()
                kind = ChunkKind.CHAPTER if _CHAPTER_RE.match(line) else ChunkKind.SECTION
                chunks.append(_base_chunk(meta, kind, line))
            elif art:
                flush_clause()
                flush_article()
                cur_article_no = cn_to_int(art.group(1))
                cur_article_lines = [line]
            elif cla and cur_article_no is not None:
                flush_clause()
                cur_clause_no = cn_to_int(cla.group(1))
                cur_clause_lines = [line]
                cur_article_lines.append(line)
            elif cur_article_no is not None:
                # 条内续行：同时归入当前条与当前款
                cur_article_lines.append(line)
                if cur_clause_lines:
                    cur_clause_lines.append(line)
        flush_clause()
        flush_article()
        return chunks


# ---------------------------------------------------------------------------
# 2. 附表切分器：按行带表头
# ---------------------------------------------------------------------------

class TableSplitter:
    """附表切分器：每一行数据生成一个子块，文本携带「列名=值」表头语义。

    例：``| 岗位 | 学历要求 |`` + ``| 律师助理 | 本科 |`` 产出子块
    「岗位=律师助理；学历要求=本科」，避免脱离表头后行内容不可解。
    """

    def split(self, raw: RawDoc) -> List[Chunk]:
        meta = raw.meta
        chunks: List[Chunk] = []
        for m in _TABLE_BLOCK_RE.finditer(raw.text):
            table_name = m.group(1).strip()
            rows = [r.strip() for r in m.group(2).strip().splitlines() if r.strip()]
            if len(rows) < 2:
                continue
            header = self._parse_row(rows[0])
            parent = _base_chunk(
                meta, ChunkKind.TABLE,
                f"《{meta.law_name}》附表《{table_name}》表头: " + "；".join(header),
            )
            parent.extra["table_name"] = table_name
            parent.extra["header"] = header
            row_chunks: List[str] = []
            for i, row_text in enumerate(rows[1:]):
                cells = self._parse_row(row_text)
                pairs = [f"{h}={v}" for h, v in zip(header, cells)] or cells
                child = _base_chunk(
                    meta, ChunkKind.TABLE_ROW,
                    f"附表《{table_name}》: " + "；".join(pairs),
                    parent_id=parent.chunk_id,
                )
                child.extra["row_index"] = i
                child.extra["table_name"] = table_name
                chunks.append(child)
                row_chunks.append(child.chunk_id)
            parent.child_ids = row_chunks
            chunks.append(parent)
        return chunks

    @staticmethod
    def _parse_row(row: str) -> List[str]:
        """行解析：优先制表符，其次竖线，最后连续空白。"""
        if "\t" in row:
            return [c.strip() for c in row.split("\t")]
        if "|" in row:
            return [c.strip() for c in row.strip("|").split("|")]
        return re.split(r"\s{2,}|\s+", row.strip())


# ---------------------------------------------------------------------------
# 3. 类案切分器：语义断崖
# ---------------------------------------------------------------------------

class CaseSplitter:
    """类案切分器：在相邻段落语义相似度骤降处（断崖）断开。

    真实部署用 BGE-M3 稠密向量计算相邻段余弦相似度；
    裸环境降级为字符二元组 Jaccard 相似度，接口保持一致。
    """

    def __init__(self, cliff_threshold: float = 0.18, min_len: int = 40):
        self.cliff_threshold = cliff_threshold
        self.min_len = min_len

    @staticmethod
    def _bigrams(text: str) -> set:
        t = re.sub(r"\s+", "", text)
        return {t[i:i + 2] for i in range(len(t) - 1)}

    def _similarity(self, a: str, b: str) -> float:
        ga, gb = self._bigrams(a), self._bigrams(b)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)

    def split(self, raw: RawDoc) -> List[Chunk]:
        meta = raw.meta
        paras = [p.strip() for p in re.split(r"\n{2,}", raw.text) if p.strip()]
        groups: List[List[str]] = []
        cur: List[str] = []
        for p in paras:
            if cur and self._similarity(cur[-1], p) < self.cliff_threshold:
                groups.append(cur)
                cur = [p]
            else:
                cur.append(p)
        if cur:
            groups.append(cur)
        chunks: List[Chunk] = []
        for gi, g in enumerate(groups):
            text = "\n".join(g)
            if len(re.sub(r"\s", "", text)) < self.min_len:
                continue  # 过短残段丢弃，避免噪声块
            c = _base_chunk(meta, ChunkKind.CASE_PASSAGE, text)
            c.extra["group_index"] = gi
            chunks.append(c)
        return chunks


# ---------------------------------------------------------------------------
# 4. 咨询记录切分器：按轮次
# ---------------------------------------------------------------------------

class ConsultSplitter:
    """咨询记录切分器：一问一答配对成轮次块，保留双方发言。

    仅保留咨询人问题会导致答案语境缺失；成对切分使 FAQ 短路
    与类案检索都能命中完整交互。
    """

    _TURN_RE = re.compile(r"^(咨询人|律师|用户|法务)\s*[:：]")

    def split(self, raw: RawDoc) -> List[Chunk]:
        meta = raw.meta
        lines = [ln.strip() for ln in raw.text.splitlines() if ln.strip()]
        turns: List[List[str]] = []
        cur: List[str] = []
        for line in lines:
            if self._TURN_RE.match(line):
                if cur:
                    turns.append(cur)
                cur = [line]
            elif cur:
                cur.append(line)
        if cur:
            turns.append(cur)

        chunks: List[Chunk] = []
        i = 0
        while i < len(turns):
            t = turns[i]
            # 尝试与下一轮组成 问+答 对
            pair = list(t)
            if i + 1 < len(turns) and self._is_ask(t) and self._is_answer(turns[i + 1]):
                pair.extend(turns[i + 1])
                i += 2
            else:
                i += 1
            c = _base_chunk(meta, ChunkKind.CONSULT_TURN, "\n".join(pair))
            c.extra["turn_index"] = len(chunks)
            chunks.append(c)
        return chunks

    @staticmethod
    def _is_ask(turn: List[str]) -> bool:
        return bool(turn) and turn[0].startswith(("咨询人", "用户"))

    @staticmethod
    def _is_answer(turn: List[str]) -> bool:
        return bool(turn) and turn[0].startswith(("律师", "法务"))


# ---------------------------------------------------------------------------
# 路由入口
# ---------------------------------------------------------------------------

_SPLITTERS = {
    "regulation": RegulationSplitter,
    "table": TableSplitter,
    "case": CaseSplitter,
    "consult": ConsultSplitter,
}


def detect_doc_type(raw: RawDoc) -> str:
    """按元数据或正文特征推断文档类型。"""
    declared = raw.meta.extra.get("doc_type")
    if declared in _SPLITTERS:
        return declared
    if raw.meta.sensitivity == Sensitivity.CASE:
        return "consult" if raw.meta.extra.get("is_consult") else "case"
    if re.search(r"^第[零一二三四五六七八九十百千0-9]+条", raw.text, flags=re.M):
        return "regulation"
    if _TABLE_BLOCK_RE.search(raw.text):
        return "table"
    if re.search(r"^(咨询人|用户)\s*[:：]", raw.text, flags=re.M):
        return "consult"
    return "case"


def split_document(raw: RawDoc) -> List[Chunk]:
    """切分单文档：法规切分器始终先行，附表块独立由 TableSplitter 处理。"""
    doc_type = detect_doc_type(raw)
    chunks = _SPLITTERS[doc_type]().split(raw)
    if doc_type != "table" and raw.tables:
        # 法规正文内嵌附表：附表行块与正文块并存
        table_raw = RawDoc(meta=raw.meta, text="\n".join(raw.tables))
        chunks.extend(TableSplitter().split(table_raw))
    for c in chunks:
        c.extra.setdefault("doc_type", doc_type)
    return chunks
