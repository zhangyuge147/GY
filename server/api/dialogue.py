# -*- coding: utf-8 -*-
"""
对话 API（任务 A2 / A3 / A3-1 / A4 / 6.2 对话模块）：
  GET  /api/deeds/{hero_id}        事迹卡列表（直接读史料库，不经 LLM）
  POST /api/dialogue/narrate       讲解生成：TTS 音频先行返回，视频后台异步合成
讲解画面固定用库内英雄标准像（A3-1），与用户上传照片解耦。
"""
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
import tasks

router = APIRouter(prefix="/api", tags=["dialogue"])

_DEEDS = None


def _deeds() -> dict:
    global _DEEDS
    if _DEEDS is None:
        _DEEDS = config.load_deeds()
    return _DEEDS


@router.get("/deeds/{hero_id}")
def deeds(hero_id: str):
    h = config.HEROES.get(hero_id)
    if not h:
        raise HTTPException(404, "未收录该英雄")
    d = _deeds().get(hero_id)
    if not d:
        raise HTTPException(404, "该英雄暂无史料")
    return {"hero_id": hero_id, "name": h["name"], "info": d.get("info", ""),
            "deeds": d["deeds"], "sources": d["sources"]}


class NarrateIn(BaseModel):
    hero_id: str
    deed_title: str | None = None
    item_text: str | None = None


@router.post("/dialogue/narrate")
def narrate(body: NarrateIn):
    h = config.HEROES.get(body.hero_id)
    if not h:
        raise HTTPException(404, "未收录该英雄")
    portrait = config.hero_portrait_path(body.hero_id)
    if not portrait:
        raise HTTPException(400, "该英雄暂无库内标准像，暂不支持讲解")

    text = config.build_narration(body.hero_id, body.deed_title, body.item_text)
    config.load_env()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 音频先行：同步合成 TTS wav
    import tts as tts_mod
    ts = time.strftime("%Y%m%d_%H%M%S")
    audio = config.OUT_DIR / f"{body.hero_id}_{ts}_narrate.wav"
    tts_mod.synthesize(text, audio)

    # 2) 视频后台异步合成（讲解画面用标准像 + 开端照片）
    sources = _deeds().get(body.hero_id, {}).get("sources", [])

    def _make_video():
        import video as video_mod
        out = config.OUT_DIR / f"{body.hero_id}_{ts}_narrate.mp4"
        credits = ["史料出处"] + sources + [
            "", "本片开篇画面由 AI 生成；讲解语音与影像均为 AI 演绎，仅供参考。"]
        video_mod.render(str(config.OPENING), str(portrait), str(audio), str(out),
                         captions=text, credits=credits)
        return {"video_url": f"/out/{out.name}"}

    task_id = tasks.submit(_make_video)
    return {"audio_url": f"/out/{audio.name}", "captions": text,
            "task_id": task_id, "status": "queued"}