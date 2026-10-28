# -*- coding: utf-8 -*-
"""
问答 API（任务 4.7 / 6.2 问答模块，B5 追问兜底）：
  POST /api/rag/ask    自由提问 → 检索 + DeepSeek 生成 → 答案 + 出处 + 命中史料
复用 knowledge/rag.py 的检索 / prompt / 生成 / 留档逻辑，进程内懒加载一次索引。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
import rag as rag_mod
import search as search_mod

router = APIRouter(prefix="/api", tags=["rag"])

# 进程内缓存：加载一次 FAISS 索引 + 嵌入模型（较重）
_RUNTIME = None

# 素材库检索 top1 相似度低于此值 => 视为「无强相关」，触发受限联网兜底
WEB_THRESHOLD = 0.5

WEB_SYSTEM = (
    "你是「光影重生」红色历史影像修复平台的问答助手。"
    "以下内容来自联网检索（仅权威新闻 / 党史网站），可能未经人工核验。"
    "要求：1. 必须在回答开头明确写「以下内容来自联网检索，未经人工核验」；"
    "2. 仅依据下方联网检索结果作答，引用时标注来源编号 [1][2]；"
    "3. 来源不足或不确定时如实说明，绝不编造；"
    "4. 语言通俗、客观、克制，面向青少年。"
)

KNOWLEDGE_SYSTEM = (
    "你是「光影重生」红色历史影像修复平台的问答助手。"
    "你无法联网核验，将基于你自身训练知识回答。"
    "要求：1. 必须在回答开头明确写「以下内容由 AI 基于自身知识生成，未经权威史料核验，仅供了解参考」；"
    "2. 客观、克制、不编造；涉及时间 / 地点 / 言论等关键史实若不确定，如实说明；"
    "3. 结尾建议用户以权威党史资料（如人民网、共产党员网）为准；"
    "4. 语言通俗，面向青少年。"
)


class AskIn(BaseModel):
    question: str
    topk: int = 3


def _ensure_loaded():
    global _RUNTIME
    if _RUNTIME is None:
        config.load_env()
        _RUNTIME = rag_mod.load_index()
    return _RUNTIME


def _web_answer(question: str, web_sources: list) -> str | None:
    """用受限联网结果 + DeepSeek 生成带「未核验」标注的回答；失败返回 None。"""
    lines = ["联网检索结果（来自权威新闻 / 党史网站，未经人工核验）："]
    for i, s in enumerate(web_sources, 1):
        lines.append(f"[{i}] {s['title']}（{s['url']}）\n{s['snippet']}")
    lines.append("")
    lines.append(f"问题：{question}")
    prompt = "\n".join(lines)
    return rag_mod.deepseek_generate(prompt, system=WEB_SYSTEM)


def _knowledge_answer(question: str) -> str | None:
    """联网不可达时用 DeepSeek 自身知识兜底，明确标注「未经核验」；失败返回 None。"""
    return rag_mod.deepseek_generate(f"问题：{question}", system=KNOWLEDGE_SYSTEM)


@router.post("/rag/ask")
def ask(body: AskIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "问题不能为空")

    index, chunks, model = _ensure_loaded()
    retrieved = rag_mod.retrieve(q, index, chunks, model, topk=max(1, body.topk))
    best = retrieved[0][1] if retrieved else -1.0

    # 素材库检索不到（相似度过低）→ 受限联网兜底 → 知识兜底
    if best < WEB_THRESHOLD:
        config.load_env()
        web_sources = search_mod.web_search(q)
        if web_sources:
            answer = _web_answer(q, web_sources)
            if answer:
                return {"question": q, "answer": answer,
                        "citations": {}, "retrieved": [],
                        "web": True, "web_sources": web_sources, "fallback": False}
        # 联网不可达 / 无权威结果 → DeepSeek 自身知识兜底（标注未核验）
        answer = _knowledge_answer(q)
        if answer:
            return {"question": q, "answer": answer,
                    "citations": {}, "retrieved": [],
                    "web": False, "web_sources": [], "fallback": True}

    if not retrieved:
        return {"question": q,
                "answer": "根据现有史料，暂不足以回答这个问题，欢迎换个问题再聊。",
                "citations": {}, "retrieved": [], "web": False, "web_sources": [],
                "fallback": False}

    src_map = rag_mod._dedupe_sources(retrieved)
    prompt = rag_mod.build_prompt(q, retrieved, src_map)
    answer = rag_mod.deepseek_generate(prompt)
    if answer is None:
        # 生成失败（未配置 key 或偶发网络异常）时仍返回命中的史料供前端展示
        answer = "（DeepSeek 调用失败或未配置，仅展示检索到的史料）"

    rag_mod.log_qa(q, retrieved, src_map, answer)
    return {
        "question": q,
        "answer": answer,
        "citations": src_map,
        "retrieved": [
            {"person": c["person"], "section": c["section"],
             "text": c["text"], "score": round(s, 4)}
            for c, s in retrieved],
        "web": False,
        "web_sources": [],
        "fallback": False,
    }