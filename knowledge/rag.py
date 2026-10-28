# -*- coding: utf-8 -*-
"""
任务 4.4 / 4.5 / 4.6：史料问答 RAG（检索增强问答）
=====================================================================
流程：提问 → bge 向量化 → FAISS 检索 top-k 史料摘要 chunk → 拼 prompt（含
硬规则 system prompt）→ DeepSeek 生成答案 → 输出答案 + [1][2] 引用编号 + 出处卡片，
并把每次问答写入 qa_log.jsonl 留档。

硬约束（项目 memory）：
  - 史料知识库只存结构事实摘要、标注出处，不搬原文；答案据此回指 [n]。
  - 严禁编造史实：检索材料不足时必须明说"资料不足"，不得脑补。
  - 人物评价需客观、符合党史权威表述，不虚构细节、不传播传闻。

LLM 可插拔：未配置 DeepSeek 密钥时，deepseek_generate 返回 None，主流程仍输出
"检索到的史料 + 出处"，便于先验证检索质量（4.7 里程碑的检索侧）。键位读自 .env：
  DEEPSEEK_API_KEY（必填）/ DEEPSEEK_MODEL（默认 deepseek-chat）/ DEEPSEEK_BASE_URL
  接口为 OpenAI 兼容：POST {BASE_URL}/chat/completions + Bearer Token。

用法（guangying 环境，在 guangying/ 目录运行）：
  python knowledge/rag.py --question "焦裕禄的主要事迹是什么"
  python knowledge/rag.py -q "杨靖宇是怎么牺牲的" --topk 3
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent          # knowledge/
INDEX_DIR = ROOT / "index"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"   # 与 build_index 保持一致；换 large 时两处同步改
ENV_FILE = ROOT.parent / ".env"                 # guangying/.env
LOG_FILE = ROOT / "qa_log.jsonl"

SYSTEM_PROMPT = """你是「光影重生」红色历史影像修复项目的史料问答助手，任务是用权威、客观、克制的方式，为用户（青少年红色教育受众）讲解历史人物事迹。

必须遵守以下硬规则：
1. 只依据下方"参考史料"作答，不引用参考史料之外的信息，不编造、不臆测任何时间、地点、言论、细节。
2. 涉及人物评价必须客观、符合党史权威表述；不虚构对话或心理活动。
3. 回答问题中引用关键事实时，必须用方括号标注对应史料编号，如 [1]、[2]。
4. 若参考史料不足以回答该问题，直接回答"根据现有史料，暂不足以回答这个问题"，不得补全或猜测。
5. 区分"史实陈述"与"精神内涵"：史实如实转述；精神层面的感悟用简短、克制的语言，不夸大、不煽情。
6. 语言通俗易懂，面向青少年；篇幅适中，直击问题，不堆砌。
"""


def load_env(path: Path = ENV_FILE):
    """极简 .env 解析（避免额外依赖 python-dotenv）。"""
    if not path.exists():
        return
    for l in path.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if not l or l.startswith("#") or "=" not in l:
            continue
        k, _, v = l.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def load_index(index_dir: Path = INDEX_DIR, model_name: str = DEFAULT_MODEL):
    """加载 FAISS 索引 + chunk 元数据 + 嵌入模型。"""
    import faiss
    _slug = "models--" + model_name.replace("/", "--")
    if (Path.home() / ".cache" / "huggingface" / "hub" / _slug).exists():
        # 模型已缓存时跳过 hub 网络校验，弱网/离线环境可避免卡死
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "chunks.json"
    if not idx_path.exists() or not meta_path.exists():
        raise SystemExit(f"[!] 未找到索引，请先运行 build_index.py（缺 {idx_path} / {meta_path}）")

    index = faiss.read_index(str(idx_path))
    chunks = json.loads(meta_path.read_text(encoding="utf-8"))
    model = SentenceTransformer(model_name)
    return index, chunks, model


def retrieve(question, index, chunks, model, topk=3):
    """向量检索 top-k，返回 [(chunk, score), ...]，score 为 cosine=-1..1。"""
    qv = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, min(topk, len(chunks)))
    out = []
    for i, (s, cid) in enumerate(zip(scores[0], ids[0])):
        if cid < 0:
            continue
        out.append((chunks[cid], float(s)))
    return out


def _dedupe_sources(retrieved):
    """收集命中 chunk 的全部出处并去重、编号，返回 {编号: 出处}。"""
    seen, sources = [], []
    for c, _ in retrieved:
        for s in c.get("sources", []):
            if s and s not in seen:
                seen.append(s)
                sources.append(s)
    return {i + 1: s for i, s in enumerate(sources)}


def build_prompt(question, retrieved, src_map):
    lines = ["参考史料（仅限以下内容作答，并注明 [编号]）："]
    for i, (c, _) in enumerate(retrieved, 1):
        cite = ""
        # 该 chunk 的出处映射到全局编号
        ids = [n for n, s in src_map.items() if s in c.get("sources", [])]
        if ids:
            cite = "[" + ",".join(str(n) for n in ids) + "]"
        lines.append(f"[{i}]（{c['person']}｜{c['section']}）{c['text']} {cite}")
    lines.append("")
    lines.append(f"问题：{question}")
    return "\n".join(lines)


def deepseek_generate(prompt, system=SYSTEM_PROMPT):
    """DeepSeek 生成（OpenAI 兼容 HTTP 接口 + Bearer Token）。未配置密钥返回 None。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")

    try:
        import requests
    except ImportError:
        raise SystemExit("[!] 缺 requests，请先：pip install requests")

    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1024,
                "stream": False,
            },
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        # 网络抖动 / SSL 重置 / 超时：降级为「仅展示检索结果」，不让整个问答接口 500
        print(f"[rag] DeepSeek 请求失败，降级为仅展示检索结果：{e}", flush=True)
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek 返回 {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(
            f"DeepSeek 返回结构异常: {json.dumps(data, ensure_ascii=False)[:300]}")


def log_qa(question, retrieved, src_map, answer, path: Path = LOG_FILE):
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "hit_chunks": [
            {"person": c["person"], "section": c["section"],
             "text": c["text"], "score": round(s, 4)}
            for c, s in retrieved],
        "citations": src_map,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def render(result: dict) -> str:
    """把结果排版为终端可读文本。"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"问：{result['question']}")
    lines.append("=" * 60)
    lines.append(result["answer"])
    if result["citations"]:
        lines.append("")
        lines.append("出处：")
        for n, s in result["citations"].items():
            lines.append(f"  [{n}] {s}")
    lines.append("")
    lines.append("命中史料：")
    for c, s in result["retrieved"]:
        lines.append(f"  · ({c['person']}｜{c['section']}) {c['text'][:60]}… "
                     f"[sim={s:.3f}]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="史料问答 RAG（检索 + DeepSeek 生成）")
    ap.add_argument("-q", "--question", type=str, required=True, help="提问")
    ap.add_argument("--topk", type=int, default=3, help="检索条数（默认 3）")
    ap.add_argument("--no-llm", action="store_true",
                    help="跳过 LLM，仅输出检索到的史料与出处（调试检索用）")
    args = ap.parse_args()

    load_env()
    index, chunks, model = load_index()
    retrieved = retrieve(args.question, index, chunks, model, topk=args.topk)
    if not retrieved:
        print("未检索到相关史料。")
        return
    src_map = _dedupe_sources(retrieved)

    if args.no_llm:
        answer = "（LLM 未调用：--no-llm 仅展示检索结果）"
    else:
        prompt = build_prompt(args.question, retrieved, src_map)
        answer = deepseek_generate(prompt)
        if answer is None:
            answer = ("（DeepSeek 未配置：请先在 .env 填写 DEEPSEEK_API_KEY，"
                      "或加 --no-llm 仅看检索）")

    result = {
        "question": args.question,
        "answer": answer,
        "retrieved": retrieved,
        "citations": src_map,
    }
    print(render(result))
    log_qa(args.question, retrieved, src_map, answer)
    print(f"\n[log] 已留档 → {LOG_FILE}")


if __name__ == "__main__":
    main()