# -*- coding: utf-8 -*-
"""
对话 API（多轮上下文 + 人设切换 + DeepSeek 流式 SSE）。

POST /api/chat
  body: { "persona": "general" | "<hero_id>", "messages": [{"role","content"}, ...] }
  返回: text/event-stream，逐段 `data: {"delta": "..."}`，结束 `data: {"done": true}`；
        出错 `data: {"error": "..."}`。

人设：
  - general          通用红色历史问答助手
  - <hero_id>        对应英雄（见 config.HEROES 中 has_portrait 者）第一人称口吻
"""
import json
import os
from typing import List

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config

router = APIRouter(prefix="/api", tags=["chat"])


def _corpus_roster() -> str:
    """史料检索页收录人物名单（corpus 文件名实时读盘，无缓存）。

    实时读取的意义：知识库语料更新后无需重启 server，引导名单即随最新
    收录变化（对比 rag 接口的 _RUNTIME 索引缓存——那个更新后必须重启）。
    """
    try:
        return "、".join(sorted(p.stem for p in config.CORPUS_DIR.glob("*.txt")))
    except Exception:
        return ""


def _general_system() -> str:
    """通用助手系统提示词（L46，2026-10-28 江姐实测反馈修复）。

    原提示词让 LLM 无差别建议"到史料检索页查证"，但 LLM 不知道平台收录
    情况：未收录人物也这么引导就是死链（收录人物语料刚入库、索引缓存未
    刷新时同样查不到）。修复：注入实时收录名单——名单内人物指向平台史
    料检索页，名单外指向人民网、共产党员网等权威党史资料，不指向平台页。
    """
    roster = _corpus_roster()
    if roster:
        guide = (
            f"平台「史料检索」页收录了以下人物的权威史料：{roster}。"
            "若用户所问人物在上述名单中，可建议到「史料检索」页查证权威出处；"
            "不在名单中的人物，请建议以人民网、共产党员网等权威党史资料为准，"
            "不要指向平台「史料检索」页。"
        )
    else:
        guide = "对不确定的内容，建议用户以人民网、共产党员网等权威党史资料为准。"
    return (
        "你是「光影重生」红色历史影像修复平台的智能问答助手。"
        "请用简洁、庄重、通俗的语言回答用户关于革命英雄、红色历史与影像修复的问题；"
        "回答必须符合权威史实，不编造、不夸大；对不确定的内容要明确说明。"
        + guide
    )


class ChatMsg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    persona: str = "general"
    messages: List[ChatMsg]


def _hero_system(hero_id: str) -> str:
    h = config.HEROES.get(hero_id)
    return (
        f"你是{h['name']}（{h['years']}，{h['title']}）。"
        f"请以第一人称、符合史实的口吻与用户对话，话题围绕你的生平与革命经历。"
        f"回答需克制庄重、合乎历史事实，不编造细节；对不确定的内容要说明并建议查证史料。"
    )


def _system_prompt(persona: str) -> str:
    if persona in config.HEROES and config.HEROES[persona].get("has_portrait"):
        return _hero_system(persona)
    return _general_system()


def _stream_chat(system: str, messages: List[ChatMsg]):
    """生成 SSE 流。每段 yield `data: {json}\n\n`。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        yield _sse({"error": "DeepSeek 未配置，请检查 .env"})
        return

    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    payload = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content} for m in messages
    ]

    try:
        with requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": payload,
                  "temperature": 0.7, "max_tokens": 1024, "stream": True},
            timeout=60,
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                yield _sse({"error": f"DeepSeek 返回 {resp.status_code}"})
                return
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield _sse({"delta": delta})
    except requests.exceptions.RequestException as e:
        yield _sse({"error": f"网络异常：{e}"})

    yield _sse({"done": True})


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@router.post("/chat")
def chat(body: ChatIn):
    if not body.messages:
        raise HTTPException(400, "消息不能为空")
    config.load_env()
    system = _system_prompt(body.persona)
    return StreamingResponse(
        _stream_chat(system, body.messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )