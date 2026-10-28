# -*- coding: utf-8 -*-
"""
任务 4.3：史料知识库索引构建（bge-large-zh-v1.5 嵌入 + FAISS）
=====================================================================
读取 knowledge/corpus/ 下"一人一档"的结构化史料摘要（.txt/md），解析出
人物名 / 章节 / 正文 / 引用出处，按条目切分后送入本地中文嵌入模型
bge-large-zh-v1.5 生成向量，写入 FAISS 内积索引 + 元数据 JSON，供 rag.py 检索。

硬约束（详见项目 memory）：
  史料知识库仅存储"结构化事实摘要"，每条标注引用出处，不全文搬运原文。
  本脚本只读 corpus/ 内已审核的摘要，生成索引时把"引用出处"作为元数据随 chunk 保留，
  确保 RAG 答案能回指 [1][2] 出处。

模型下载：首次运行需联网拉取 BAAI/bge-large-zh-v1.5（约 1.3GB）。国内建议走
  huggingface 镜像：设置环境变量 HF_ENDPOINT=https://hf-mirror.com 后再跑，
  可显著提速并规避断流。

用法（guangying 环境，在 guangying/ 目录运行）：
  python knowledge/build_index.py
  python knowledge/build_index.py --model BAAI/bge-large-zh-v1.5 --topk 3
产出：knowledge/index/faiss.index  +  knowledge/index/chunks.json
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent          # knowledge/
CORPUS = ROOT / "corpus"
INDEX_DIR = ROOT / "index"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"   # 默认小模型：~100MB，够本语料检索；网络允许可换 bge-large-zh-v1.5
MAX_CHUNK_LEN = 256          # 单条 chunk 超过此长度再按句切分
MIN_CHUNK_LEN = 20           # 短于此长度的碎片并入上一条，避免噪音片段


def parse_summary(text: str) -> dict:
    """解析一篇史料摘要为 {person, sections:[(title, content)], sources:[...]}。
    约定格式：# <人物>人物摘要 / ## 章节名 / ## 引用出处（编号列表）。"""
    lines = [l.rstrip() for l in text.splitlines()]
    lines = [l for l in lines if l.strip()]

    person = None
    sections = []          # [(title, [content_lines])]
    sources = []
    cur_title = None
    cur_lines = []

    def flush():
        nonlocal cur_title, cur_lines
        if cur_title is not None:
            sections.append((cur_title, cur_lines))
        cur_title, cur_lines = None, []

    for l in lines:
        if l.startswith("# ") and person is None:
            person = l.lstrip("# ").strip()
            person = re.sub(r"人物摘要$", "", person).strip()
        elif l.startswith("## "):
            flush()
            cur_title = l.lstrip("# ").strip()
        elif cur_title == "引用出处":
            m = re.match(r"^\s*\d+[.、]\s*(.*)$", l)
            if m:
                sources.append(m.group(1).strip())
        else:
            cur_lines.append(l)
    flush()

    person = person or "未命名人物"
    return {"person": person, "sections": sections, "sources": sources}


def chunkize(parsed: dict):
    """把解析结果切成带元数据的 chunk 列表。
    章节标题单独保留为结构化字段，正文按条目/句切分，每条附 person/section/sources。"""
    person = parsed["person"]
    sources = parsed["sources"]
    chunks = []

    for sec_title, sec_lines in parsed["sections"]:
        if sec_title == "引用出处":
            continue
        # 收集该章节的正文条目（bullet 或普通行）
        entries = []
        buf = []
        for l in sec_lines:
            s = re.sub(r"^\s*[-*•·]\s*", "", l).strip()
            s = re.sub(r"^\s*\d+[.、]\s*", "", s).strip()
            if not s:
                continue
            buf.append(s)
        if not buf:
            continue
        body = "\n".join(buf)
        # 超长章节按句切分（中英文句末符）
        pieces = [body] if len(body) <= MAX_CHUNK_LEN else \
            re.split(r"(?<=[。！？；!?;])\s*", body)
        acc = []
        acc_len = 0
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if acc_len and acc_len + len(p) > MAX_CHUNK_LEN:
                chunks.append(_mk_chunk(person, sec_title, " ".join(acc), sources))
                acc, acc_len = [], 0
            acc.append(p)
            acc_len += len(p)
        if acc:
            chunks.append(_mk_chunk(person, sec_title, " ".join(acc), sources))

    # 丢弃过短碎片
    return [c for c in chunks if len(c["text"]) >= MIN_CHUNK_LEN]


def _mk_chunk(person, section, text, sources):
    return {
        "person": person,
        "section": section,
        "text": text,
        "sources": list(sources),
        # 向量化时带上人物名作为语义前缀，提升"这个人是谁"类查询的召回
        "embed": f"{person}｜{section}｜{text}",
    }


def build(corpus=CORPUS, index_dir=INDEX_DIR, model_name=DEFAULT_MODEL):
    _slug = "models--" + model_name.replace("/", "--")
    if (Path.home() / ".cache" / "huggingface" / "hub" / _slug).exists():
        # 模型已缓存时跳过 hub 网络校验，弱网/离线环境可避免卡死（首次下载不受影响）
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from sentence_transformers import SentenceTransformer

    corpus, index_dir = Path(corpus), Path(index_dir)
    files = sorted(corpus.glob("*.txt")) + sorted(corpus.glob("*.md"))
    if not files:
        raise SystemExit(f"[!] corpus 目录无摘要文件：{corpus}")

    all_chunks = []
    for f in files:
        parsed = parse_summary(f.read_text(encoding="utf-8"))
        cs = chunkize(parsed)
        print(f"[parse] {f.name}: {parsed['person']}  {len(cs)} chunks, "
              f"{len(parsed['sources'])} 出处")
        all_chunks.extend(cs)

    if not all_chunks:
        raise SystemExit("[!] 未解析出任何 chunk，请检查摘要格式")

    print(f"[embed] 加载模型 {model_name} ...")
    model = SentenceTransformer(model_name)
    texts = [c["embed"] for c in all_chunks]
    print(f"[embed] 编码 {len(texts)} 条 ...")
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False).astype("float32")

    import faiss
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)      # 余弦相似度等价（向量已归一化）
    index.add(vecs)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    # 元数据不含向量，保留原文本与出处
    meta = [{"person": c["person"], "section": c["section"],
             "text": c["text"], "sources": c["sources"]} for c in all_chunks]
    (index_dir / "chunks.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] {len(all_chunks)} 条 chunk → {index_dir}")
    print(f"       faiss.index  dim={dim}  chunks.json 已写入")


def main():
    ap = argparse.ArgumentParser(description="史料知识库索引构建（bge + FAISS）")
    ap.add_argument("--corpus", type=str, default=str(CORPUS))
    ap.add_argument("--index", type=str, default=str(INDEX_DIR))
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = ap.parse_args()
    build(corpus=args.corpus, index_dir=args.index, model_name=args.model)


if __name__ == "__main__":
    main()