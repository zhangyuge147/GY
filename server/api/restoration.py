# -*- coding: utf-8 -*-
"""
修复 API（任务 A5 / 6.2 修复模块）：
  GET  /api/heroes                     英雄列表（预置照 + 标准像 + 修复中间图是否就绪）
  GET  /api/heroes/{hero_id}           单个英雄详情
  GET  /api/heroes/{hero_id}/frames    六步修复中间图 + 前后对照（供 6.4 / 6.4b）
  POST /api/restore                    自由上传图片 → 异步跑修复流水线（GPU，走子进程）
  GET  /api/restore/{task_id}          异步任务状态轮询（A4）
  POST /api/story                      照片背后的故事（仅史料库收录人物；未收录不编造）
"""
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import config
import tasks
import rag as rag_mod              # knowledge/rag.py（config 已把 knowledge 加入 sys.path）
from api import rag as rag_api     # 复用问答模块的索引缓存与检索阈值

router = APIRouter(prefix="/api", tags=["restoration"])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _frames_for_stem(stem: str) -> list[dict]:
    """从任意 steps/<stem>/ 目录读取六步中间图，返回 URL 列表（供上传修复结果复用）。"""
    d = config.STEPS_DIR / stem

    def _u(f):
        return f"/steps/{stem}/{f}" if (d / f).exists() else None

    frames = [{"key": "input", "name": "原图", "url": _u(config.INPUT_FILE)}]
    for key in config.STEP_ORDER:
        frames.append({"key": key, "name": config.STEP_NAMES[key],
                       "url": _u(config.STEP_FILES[key])})
    frames.append({"key": "final", "name": "成片", "url": _u(config.FINAL_FILE)})
    return frames


def _run_pipeline(image_path: Path, hero_id: str | None, prompt: str | None) -> dict:
    """子进程调用 restoration/pipeline.py（必须在 restoration/ 目录下运行）。"""
    cmd = [sys.executable, "pipeline.py", "--input", str(image_path.resolve())]
    if hero_id and hero_id in config.HEROES:
        cmd += ["--calib", config.HEROES[hero_id]["name"]]
    if prompt and prompt.strip():
        # L45 修复偏好：仅做关键词匹配，60 字上限截断防滥用；后端不解析语义
        cmd += ["--prompt", prompt.strip()[:60]]
    r = subprocess.run(cmd, cwd=str(config.RESTORE), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-800:])
    stem = image_path.stem
    final = config.STEPS_DIR / stem / config.FINAL_FILE
    if not final.exists():
        raise RuntimeError(f"修复未产出结果：{final}")
    return {"stem": stem,
            "final_url": f"/steps/{stem}/{config.FINAL_FILE}",
            "frames": _frames_for_stem(stem)}


@router.get("/heroes")
def list_heroes():
    out = []
    for hid, h in config.HEROES.items():
        portrait = config.hero_portrait_path(hid)
        out.append({
            "id": hid,
            "name": h["name"],
            "years": h["years"],
            "title": h["title"],
            "has_portrait": h["has_portrait"],
            "portrait_url": (f"/steps/{h['steps_dir']}/{config.FINAL_FILE}"
                             if portrait else None),
            "frames_ready": config.hero_frames(hid) is not None,
        })
    return {"heroes": out}


@router.get("/heroes/{hero_id}")
def get_hero(hero_id: str):
    h = config.HEROES.get(hero_id)
    if not h:
        raise HTTPException(404, "未收录该英雄")
    portrait = config.hero_portrait_path(hero_id)
    return {
        **h,
        "portrait_url": (f"/steps/{h['steps_dir']}/{config.FINAL_FILE}"
                         if portrait else None),
        "frames_ready": config.hero_frames(hero_id) is not None,
    }


@router.get("/heroes/{hero_id}/frames")
def hero_frames(hero_id: str):
    h = config.HEROES.get(hero_id)
    if not h:
        raise HTTPException(404, "未收录该英雄")
    frames = config.hero_frames(hero_id)
    if frames is None:
        raise HTTPException(404, "该英雄暂无修复中间图")
    return {"hero_id": hero_id, "name": h["name"], "frames": frames}


@router.post("/restore")
def restore(file: UploadFile = File(...), hero_id: str | None = Form(None),
            prompt: str | None = Form(None)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMG_EXTS:
        raise HTTPException(400, f"不支持的图片格式：{ext or '未知'}")
    if hero_id and hero_id not in config.HEROES:
        raise HTTPException(400, "非法身份选择")

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"upload_{time.strftime('%Y%m%d_%H%M%S')}"
    dest = config.UPLOAD_DIR / f"{stem}{ext}"
    dest.write_bytes(file.file.read())

    task_id = tasks.submit(_run_pipeline, dest, hero_id, prompt)
    return {"task_id": task_id, "status": "queued", "stem": stem}


@router.get("/restore/{task_id}")
def restore_status(task_id: str):
    t = tasks.get(task_id)
    if t is None:
        raise HTTPException(404, "未知任务")
    return t


# ---------- 照片背后的故事（仅史料库收录人物，未收录不编造） ----------

STORY_SYSTEM = (
    "你是「光影重生」红色历史影像修复平台的照片讲述者。"
    "用户上传了一张老照片，并告诉我们照片上的人物是谁。"
    "请依据下方参考史料，面向用户讲述这位人物的事迹与精神。"
    "硬规则："
    "1. 只能依据下方参考史料讲述，不得添加史料之外的任何情节、对话或心理描写；"
    "2. 开头第一句注明「以下内容依据平台史料库整理，由 AI 讲述」；"
    "3. 口吻温厚、克制（可如「您照片上的这位，是……」），面向青少年与家庭纪念场景；"
    "4. 关键事实标注 [n] 引用；"
    "5. 史料不足以支撑时如实说明，不得补全或猜测。"
)


class StoryIn(BaseModel):
    name: str            # 照片人物姓名
    task_id: str | None = None   # 关联的修复任务（预留，不强校验）


@router.post("/story")
def photo_story(body: StoryIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "请填写照片人物姓名")

    # 1) 查史料库（复用 RAG 索引，进程内缓存一次）
    index, chunks, model = rag_api._ensure_loaded()
    retrieved = rag_mod.retrieve(f"{name} 生平事迹", index, chunks, model, topk=4)

    # 2) 双重校验，防止「我爷爷」之类的泛称误命中某位英雄：
    #    a. 命中 chunk 的人物名须与用户输入对得上（含称谓如「杨靖宇将军」）；
    #    b. top1 相似度须过阈值（与 /api/rag/ask 的兜底阈值一致）。
    related = [(c, s) for c, s in retrieved
               if c["person"] in name or (len(name) >= 2 and name in c["person"])]
    best = related[0][1] if related else -1.0
    if best < rag_api.WEB_THRESHOLD:
        return {"found": False, "name": name, "story": "", "citations": {},
                "retrieved": [],
                "message": "史料库暂未收录该人物，平台只讲有史料依据的故事，不编造内容。"}

    # 3) 命中 → 基于史料讲述（DeepSeek 仅组织语言，不创作内容）
    src_map = rag_mod._dedupe_sources(related)
    person = related[0][0]["person"]
    prompt = rag_mod.build_prompt(f"请介绍 {person} 的生平事迹与精神",
                                  related, src_map)
    story = rag_mod.deepseek_generate(prompt, system=STORY_SYSTEM)
    if story is None:
        # DeepSeek 不可用：降级为直接展示史料原文（仍不编造）
        parts = [f"（{c['person']}｜{c['section']}）{c['text']}" for c, _ in related]
        story = "以下为史料库检索原文（AI 讲述暂不可用）：\n\n" + "\n\n".join(parts)

    return {"found": True, "name": name, "story": story, "citations": src_map,
            "retrieved": [{"person": c["person"], "section": c["section"],
                           "text": c["text"], "score": round(s, 4)}
                          for c, s in related]}