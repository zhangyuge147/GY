# -*- coding: utf-8 -*-
"""
阶段 6.2 后端全局配置：路径常量 + 英雄注册表 + 史料事迹解析。

设计约定：
  - 讲解画面固定用库内英雄标准像（07_dewhite.png），与用户上传照片解耦（任务 A3-1）。
  - 事迹列表直接读史料库 corpus/*.txt 的「核心事迹」小节，不经过 LLM（任务 A2）。
  - 六步修复中间图与 pipeline.py 的 STEP_ORDER / 文件名约定保持一致（供 6.4b 动画）。
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # guangying/
SERVER = ROOT / "server"
RESTORE = ROOT / "restoration"
STEPS_DIR = RESTORE / "data" / "steps"               # 修复中间图目录
CORPUS_DIR = ROOT / "knowledge" / "corpus"           # 史料摘要目录
OUT_DIR = ROOT / "out"                               # 语音/视频产物
UPLOAD_DIR = ROOT / "uploads"                        # 用户上传图
OPENING = ROOT / "opening.jpg"                       # 开端照片
ENV_FILE = ROOT / ".env"

# 让 server 能直接 import knowledge/rag.py、dialogue/tts.py、dialogue/video.py
# （这些目录不是包，靠 sys.path 注入后按模块名导入）
for _p in (ROOT / "knowledge", ROOT / "dialogue"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 六步修复步骤：与 restoration/pipeline.py STEP_ORDER 对齐
STEP_ORDER = ["deraw", "inpaint", "sr", "face", "color", "calib"]
STEP_NAMES = {
    "deraw": "去网纹", "inpaint": "划痕补全", "sr": "超分辨率",
    "face": "人脸精修", "color": "智能上色", "calib": "考据校色",
}
# 步骤编号 → 落盘文件名（pipeline 统一编号，见 pipeline.py 目录约定）
STEP_FILES = {
    "deraw": "01_deraw.png", "inpaint": "02_inpaint.png",
    "sr": "03_sr.png", "face": "04_face.png",
    "color": "05_color.png", "calib": "06_calib.png",
}
INPUT_FILE = "00_input.png"      # 修复前（原始输入）
FINAL_FILE = "07_dewhite.png"    # 修复后（最终成品）

# 英雄注册表。steps_dir 对应 restoration/data/steps/<dir>。
# 杨贵 / 焦裕禄暂缺库内标准像（任务 C2 待补），has_portrait=False，讲解功能禁用。
HEROES = {
    "jihongchang": {
        "id": "jihongchang", "name": "吉鸿昌", "years": "1895—1934",
        "title": "著名抗日英雄、中国共产党党员",
        "steps_dir": "wiki_jihongchang_01", "has_portrait": True,
    },
    "yangjingyu": {
        "id": "yangjingyu", "name": "杨靖宇", "years": "1905—1940",
        "title": "东北抗日联军主要领导人、著名抗日民族英雄",
        "steps_dir": "wiki_yangjingyu_01", "has_portrait": True,
    },
    "pengxuefeng": {
        "id": "pengxuefeng", "name": "彭雪枫", "years": "1907—1944",
        "title": "中国工农红军和新四军高级指挥员、军事家",
        "steps_dir": "wiki_pengxuefeng_01", "has_portrait": True,
    },
    "yanggui": {
        "id": "yanggui", "name": "杨贵", "years": "1928—2018",
        "title": "红旗渠总设计师",
        "steps_dir": None, "has_portrait": False,
    },
    "jiaoyulu": {
        "id": "jiaoyulu", "name": "焦裕禄", "years": "1922—1964",
        "title": "县委书记的榜样",
        "steps_dir": None, "has_portrait": False,
    },
}


def load_env(path: Path = ENV_FILE):
    """极简 .env 解析（与 rag.py / tts.py 保持同款逻辑，避免 python-dotenv 依赖）。"""
    if not path.exists():
        return
    for l in path.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if not l or l.startswith("#") or "=" not in l:
            continue
        k, _, v = l.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def hero_portrait_path(hero_id: str) -> Path | None:
    """英雄标准像绝对路径（讲解画面源，任务 A3-1）；无标准像返回 None。"""
    h = HEROES.get(hero_id)
    if not h or not h["has_portrait"] or not h["steps_dir"]:
        return None
    p = STEPS_DIR / h["steps_dir"] / FINAL_FILE
    return p if p.exists() else None


def hero_frames(hero_id: str) -> list[dict] | None:
    """六步中间图 + 前后对照，返回 [{key, name, url}]；无修复步骤返回 None。"""
    h = HEROES.get(hero_id)
    if not h or not h["steps_dir"]:
        return None
    d = STEPS_DIR / h["steps_dir"]
    if not d.exists():
        return None

    def _u(f):
        return "/steps/" + h["steps_dir"] + "/" + f if (d / f).exists() else None

    frames = [{"key": "input", "name": "原图", "url": _u(INPUT_FILE)}]
    for key in STEP_ORDER:
        frames.append({"key": key, "name": STEP_NAMES[key],
                       "url": _u(STEP_FILES[key])})
    frames.append({"key": "final", "name": "成片", "url": _u(FINAL_FILE)})
    return frames


def _parse_corpus(path: Path) -> dict:
    """解析单个 corpus/*.txt，返回 {name, info, deeds, sources}。

    结构约定（与 5 份史料摘要一致）：
      ## 基本信息        → 单行文字（下方跟一个自然段）
      ## 核心事迹：xxx   → title=xxx，旗下若干 "- bullet"
      ## 引用出处        → "1. 出处文本" 列表
    """
    text = path.read_text(encoding="utf-8")
    name = None
    info = ""
    deeds = []          # [{"title":..., "items":[{"text":...}]}]
    sources = []
    section = None
    cur_title = None
    cur_items = []

    def flush():
        nonlocal cur_title, cur_items
        if cur_title and cur_items:
            deeds.append({"title": cur_title, "items": cur_items[:]})
        cur_title, cur_items = None, []

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("# ") and not line.startswith("## "):
            name = line[2:].replace("人物摘要", "").strip()
            continue
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            if section.startswith("核心事迹"):
                # "核心事迹：察哈尔抗日与就义" → title 取冒号后段
                cur_title = section.split("：", 1)[-1].strip() or "主要事迹"
                cur_items = []
            elif section == "引用出处":
                cur_title = None
            continue
        ls = line.strip()
        if not ls:
            continue
        if section == "基本信息" and not ls.startswith("-"):
            info = ls
        elif section == "生平主要经历" and ls.startswith("-"):
            continue  # 生平不纳入事迹卡
        elif cur_title is not None and ls.startswith("-"):
            cur_items.append({"text": ls[1:].strip()})
        elif section == "引用出处" and re.match(r"^\d+\.", ls):
            sources.append(re.sub(r"^\d+\.\s*", "", ls))
    flush()

    return {"name": name, "info": info, "deeds": deeds, "sources": sources}


def load_deeds() -> dict:
    """读取全部 corpus，返回 {hero_id: {"name", "info", "deeds", "sources"}}。

    每条事迹 item 附带来源编号 source_nos（该人物「引用出处」全部条目，编号从 1 起）。
    """
    result = {}
    for p in sorted(CORPUS_DIR.glob("*.txt")):
        parsed = _parse_corpus(p)
        if not parsed["name"]:
            continue
        # 由 name 匹配 hero_id
        hero_id = None
        for hid, h in HEROES.items():
            if h["name"] == parsed["name"]:
                hero_id = hid
                break
        if hero_id is None:
            continue
        src_nos = list(range(1, len(parsed["sources"]) + 1))
        for deed in parsed["deeds"]:
            for it in deed["items"]:
                it["source_nos"] = src_nos
        result[hero_id] = parsed
    return result


def build_narration(hero_id: str, deed_title: str, item_text: str) -> str:
    """把所选英雄 + 事迹条目拼成讲解词（供 TTS + 视频字幕）。

    讲解字数克制、史实化，不虚构细节；信息来自史料摘要 + 所选事迹条目。
    """
    h = HEROES.get(hero_id, {})
    name = h.get("name", "")
    info = h.get("title", "")
    parts = [f"{name}，{info}。"]
    if deed_title:
        parts.append(f"下面是他的主要事迹——{deed_title}。")
    if item_text:
        parts.append(item_text)
    return "".join(parts)