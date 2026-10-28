# -*- coding: utf-8 -*-
"""
任务 5.2：火山引擎 TTS（豆包语音合成大模型）——文字转语音生成 wav。
============================================================
替换原计划的讯飞 TTS（用户决策：语音换火山，比赛无明确要求用讯飞）。

接口：大模型 HTTP 非流式 V1
  POST https://openspeech.bytedance.com/api/v1/tts
  认证头：Authorization: Bearer;{access_token}   ← 注意用分号「;」分隔，不是空格
  请求体：
    app    = {appid, token, cluster="volcano_tts"}   （token 为无鉴权作用的占位，可传 access_token）
    user   = {uid}
    audio  = {voice_type, encoding="wav", speed_ratio}
    request= {reqid=uuid, text, operation="query"}
  返回：JSON，data 字段为 base64 编码的音频；code==3000 表示成功。

硬约束（项目 memory）：
  - 使用通用庄重演绎音色，不复刻真人声音，规避伦理风险。
  - 文本来源为史料摘要，禁止编造时间/地点/言论/细节。

文本长度限制：单次 request.text ≤ 1024 字节（约 300 汉字），本脚本自动按句切分、
逐段合成后拼接为单个 wav，无需担心较长脚本超限。

用法（guangying 环境，在 guangying/ 目录运行）：
  python dialogue/tts.py --text "焦裕禄同志是县委书记的榜样。" --out out/demo.wav
  python dialogue/tts.py --file script.txt --out out/demo.wav
"""
import argparse
import base64
import io
import json
import os
import re
import uuid
import wave
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent          # dialogue/
ENV_FILE = ROOT.parent / ".env"                 # guangying/.env
API_URL = "https://openspeech.bytedance.com/api/v1/tts"
DEFAULT_VOICE = "zh_male_M392_conversation_wvae_bigtts"   # 通用男声（示例音色）
DEFAULT_CLUSTER = "volcano_tts"
MAX_CHARS = 300


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


def split_text(text: str, max_chars: int = MAX_CHARS):
    """按句末标点切分，再贪心合并到 ≤ max_chars，避免超长请求报错。"""
    text = text.replace("\n", "。")
    parts = [p for p in re.split(r"(?<=[。！？!?；;])", text) if p.strip()]
    segs, cur = [], ""
    for p in parts:
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            if cur:
                segs.append(cur)
            while len(p) > max_chars:          # 单句超长才硬切
                segs.append(p[:max_chars])
                p = p[max_chars:]
            cur = p
    if cur:
        segs.append(cur)
    return segs or [text]


def _synth_segment(seg: str, voice_type: str, speed: float, app_id: str,
                   access_token: str, cluster: str) -> bytes:
    """合成单段文本，返回该段的完整 WAV 字节。"""
    body = {
        "app": {"appid": app_id, "token": access_token, "cluster": cluster},
        "user": {"uid": "guangying"},
        "audio": {"voice_type": voice_type, "encoding": "wav",
                  "speed_ratio": speed},
        "request": {"reqid": str(uuid.uuid4()), "text": seg,
                    "operation": "query"},
    }
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer;{access_token}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"火山 TTS HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if data.get("code") != 3000:
        raise RuntimeError(
            f"火山 TTS code={data.get('code')} message={data.get('message')}: "
            f"{json.dumps(data, ensure_ascii=False)[:300]}")
    return base64.b64decode(data["data"])


def synthesize(text: str, out_path, voice_type=None, speed: float = 1.0):
    """主流程：切分 → 逐段合成 → 拼接为单个 wav，返回输出路径。"""
    load_env()
    app_id = os.environ.get("VOLC_APP_ID", "").strip()
    access_token = os.environ.get("VOLC_ACCESS_TOKEN", "").strip()
    if not app_id or not access_token:
        raise SystemExit("[!] 请在 .env 填写 VOLC_APP_ID / VOLC_ACCESS_TOKEN")
    voice = (voice_type or os.environ.get("VOLC_VOICE_TYPE", "").strip()
             or DEFAULT_VOICE)
    cluster = os.environ.get("VOLC_CLUSTER", DEFAULT_CLUSTER).strip()

    segs = split_text(text)
    params, frames = None, []
    for seg in segs:
        wav_bytes = _synth_segment(seg, voice, speed, app_id, access_token,
                                   cluster)
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    if params is None:
        raise RuntimeError("未合成任何音频")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setparams(params)
        for f in frames:
            w.writeframes(f)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="火山引擎 TTS：文字转语音生成 wav")
    ap.add_argument("--text", type=str, default=None, help="直接传入合成文本")
    ap.add_argument("--file", type=str, default=None, help="从 txt 文件读取合成文本")
    ap.add_argument("--out", type=str, default="out/demo.wav", help="输出 wav 路径")
    ap.add_argument("--voice-type", type=str, default=None, help="覆盖音色代号")
    ap.add_argument("--speed", type=float, default=1.0, help="语速 [0.1,2]，默认 1.0")
    args = ap.parse_args()

    load_env()
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        ap.error("必须提供 --text 或 --file")

    out = synthesize(text, args.out, args.voice_type, args.speed)
    print(f"[ok] 已生成 → {out}（共 {len(split_text(text))} 段合成）")


if __name__ == "__main__":
    main()