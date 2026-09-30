# -*- coding: utf-8 -*-
"""知识库构建与检索：md + PDF → 章节感知分块 → 向量化 → 持久化 → 检索。"""
import io
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config as C


# ── 分块 ──────────────────────────────────────────────────────────────
def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )


def split_markdown(text: str, meta_base: dict) -> List[Document]:
    """按 Markdown 标题切出小节，标题层级写入 metadata.section，再二次分块。"""
    lines = text.split("\n")
    sections, cur_path, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" / ".join(cur_path), body))

    for ln in lines:
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            flush()
            buf = []
            level, title = len(m.group(1)), m.group(2).strip()
            cur_path = cur_path[: level - 1] + [title]
        buf.append(ln)
    flush()

    splitter = _make_splitter()
    docs: List[Document] = []
    for section, body in sections:
        for chunk in splitter.split_text(body):
            if len(chunk.strip()) < 30:
                continue
            meta = dict(meta_base)
            meta["section"] = section
            docs.append(Document(page_content=chunk, metadata=meta))
    return docs


_PDF_NOISE = (
    re.compile(r"^FDE 全景研究报告：.*$"),
    re.compile(r"^基于公开资料的系统研究 · 2026 年 9 月\s*第 \d+ 页$"),
)
_CHAPTER = re.compile(r"^(第[一二三四五六七八九十百\d]+章[·.．]?\s*.*)$")


def split_pdf(pdf_path: Path, meta_base: dict) -> List[Document]:
    import fitz  # pymupdf
    doc = fitz.open(str(pdf_path))
    chapter = "前言"
    pages = []
    for i, page in enumerate(doc):
        raw = page.get_text("text") or ""
        kept = []
        for ln in raw.split("\n"):
            ln = ln.rstrip()
            if any(p.match(ln) for p in _PDF_NOISE):
                continue
            cm = _CHAPTER.match(ln.strip())
            if cm:
                chapter = cm.group(1).strip()
            kept.append(ln)
        pages.append("\n".join(kept).strip())
    doc.close()

    splitter = _make_splitter()
    docs: List[Document] = []
    for i, text in enumerate(pages):
        if len(text) < 30:
            continue
        for chunk in splitter.split_text(text):
            if len(chunk.strip()) < 30:
                continue
            meta = dict(meta_base)
            meta["section"] = chapter
            meta["page"] = i + 1
            docs.append(Document(page_content=chunk, metadata=meta))
    return docs


# ── 构建 / 加载 ───────────────────────────────────────────────────────
def collect_documents() -> List[Document]:
    docs: List[Document] = []
    if C.SOP_PATH.exists():
        text = io.open(str(C.SOP_PATH), encoding="utf-8").read()
        docs += split_markdown(text, {
            "doc": "FDE课程SOP（关键经验与客户洞察·项目全生命周期）",
            "source": C.SOP_PATH.name,
        })
    else:
        print(f"[警告] 找不到 SOP 文档：{C.SOP_PATH}")

    if C.COURSE_DIR.exists():
        for f in sorted(C.COURSE_DIR.glob("*.md")):
            text = io.open(str(f), encoding="utf-8").read()
            docs += split_markdown(text, {
                "doc": f"FDE实战课·{f.stem}",
                "source": f.name,
            })
    else:
        print(f"[警告] 找不到课程目录：{C.COURSE_DIR}")

    if C.REPORT_MD_DIR.exists() and any(C.REPORT_MD_DIR.glob("*.md")):
        for f in sorted(C.REPORT_MD_DIR.glob("*.md")):
            text = io.open(str(f), encoding="utf-8").read()
            docs += split_markdown(text, {
                "doc": f"FDE全景研究报告·{f.stem}",
                "source": f.name,
            })
    elif C.REPORT_PDF.exists():
        docs += split_pdf(C.REPORT_PDF, {
            "doc": "FDE全景研究报告（模式、案例与落地操作手册）",
            "source": C.REPORT_PDF.name,
        })
    else:
        print(f"[警告] 找不到研究报告（md 目录与 PDF 均缺失）：{C.REPORT_MD_DIR}")
    return docs


def ingest(cfg, embeddings: Embeddings) -> dict:
    docs = collect_documents()
    if not docs:
        raise RuntimeError("未收集到任何知识文档，请检查 config.py 中的路径配置")
    store = InMemoryVectorStore(embeddings)
    store.add_documents(docs)
    C.VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(C.VECTORSTORE_PATH))

    per_doc = {}
    for d in docs:
        per_doc[d.metadata["doc"]] = per_doc.get(d.metadata["doc"], 0) + 1
    stats = {"chunks": len(docs), "per_doc": per_doc}
    return stats


def load_vectorstore(cfg, embeddings: Embeddings) -> InMemoryVectorStore:
    if not C.VECTORSTORE_PATH.exists():
        raise RuntimeError(
            f"向量库不存在（{C.VECTORSTORE_PATH}）。请先运行：python cli.py ingest")
    return InMemoryVectorStore.load(str(C.VECTORSTORE_PATH), embedding=embeddings)


def format_docs(docs: List[Document]) -> str:
    """检索结果 → 带编号引用的上下文文本。"""
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        where = f"{m.get('doc', '?')}｜{m.get('section', '')}"
        if m.get("page"):
            where += f"｜P{m['page']}"
        parts.append(f"[{i}] （{where}）\n{d.page_content}")
    return "\n\n".join(parts)
