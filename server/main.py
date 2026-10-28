# -*- coding: utf-8 -*-
"""
阶段 6.2 FastAPI 后端入口：修复 / 问答 / 对话三大 API + 静态资源托管。

启动（在 guangying/ 目录、guangying 环境）：
  python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parent
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import config  # noqa: E402
from api import chat, dialogue, rag, restoration  # noqa: E402

app = FastAPI(title="光影重生 · 平台后端", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(restoration.router)
app.include_router(rag.router)
app.include_router(dialogue.router)
app.include_router(chat.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/opening.jpg")
def opening():
    return FileResponse(str(config.OPENING))


# 静态资源：修复中间图 / 语音视频产物 / 用户上传
config.STEPS_DIR.mkdir(parents=True, exist_ok=True)
config.OUT_DIR.mkdir(parents=True, exist_ok=True)
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/steps", StaticFiles(directory=str(config.STEPS_DIR)), name="steps")
app.mount("/out", StaticFiles(directory=str(config.OUT_DIR)), name="out")
app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")

# 前端构建产物（web/dist）；开发阶段走 Vite dev server + proxy，此处留空不挂载
WEB_DIST = config.ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")