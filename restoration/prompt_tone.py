# -*- coding: utf-8 -*-
"""
L45 修复偏好提示词（精简版，2026-10-28）
========================================
流水线第五步"智能上色"的用户偏好层：前端 6 个预设 chip（暖调/冷调/怀旧/
鲜艳/淡雅/保持黑白）→ 后端关键词匹配 → Lab 色调后处理。定位是【个性化
偏好】，不是质量提升：照片质量（保真/清晰/去划痕）由既有链路负责。

设计约束（三层护栏，见 局限与精修清单.txt L45）：
  1. 全参数钳制：预设值本身在安全区间，代码再 clamp 兜底（|Δa/Δb|≤12、
     饱和系数 0.7~1.3、对比/明度微调幅度上限），用户提示词不可能把照片
     带向极端色彩；
  2. 空提示词/未识别 → 直通零改动：不带 --prompt 的所有任务（历史 stem、
     英雄标准像、评测基准）行为与旧版完全一致；
  3. 史实优先：本模块作用于 05_color 落盘前（色相统一之后），后续考据
     校色（--calib 局部校色）仍在其后执行，考据色（红旗/军装）天然覆盖
     用户偏好。

为什么是 Lab 后处理而非"提示词引导上色模型"：DDColor 无文本条件输入，
换扩散类文生图上色模型与"保真优先"原则冲突且有显存/速度/幻觉风险（见
L45 讨论记录）。全局一致的 Lab 偏移不破坏 L44 色相统一（同物各像素同幅
移动，相对一致性不变）。

人脸保护：色温偏移（Δa/Δb）在 Haar 人脸框内强度减半（肤色对色温敏感，
全量偏移会显病态），框缘高斯羽化平滑过渡；饱和缩放保持全图统一（幅度
温和，且后续 deglare/dewhite 仍会把肤色往正常区间拉）。无脸/检测失败时
全图统一偏移，不影响功能。

用法（pipeline 内部，亦可 CLI 单测）：
  python prompt_tone.py --input <05_color.png> --prompt 怀旧 --out out.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# ---- 预设参数（Lab 空间；钳制兜底见 _clamp_params） ----
#   da/db   a/b 通道偏移（暖=+a+b 红黄，冷=-a-b 蓝青，怀旧黄=b+）
#   sat_k   ab 幅值缩放（只放大/缩小饱和度，色相不变，不破坏 L44 统一）
#   l_gain  L 通道对比度系数增量（0.5 中心线性对比；负=压对比）
#   l_bias  L 通道明度偏移（归一化 0-1）
#   l_drop  高光压亮幅度（v2 新增）：Lab 中 L=100 白点 ab 恒 0，转 BGR
#           必裁剪回白——老照片大面积过曝白区加多少 ab 都显不出色调，
#           必须"降 L + 偏 ab"组合才可感（白窗→米黄/青灰/纸黄）。仅对
#           L>0.55 的亮部按平滑权重生效，暗部零影响。
# v2 幅度标定（2026-10-28 用户实测"选不同偏好基本没差别"后当日增强）：
#   v1 幅度对复杂场景观感偏弱（Δa+5 全局偏移在 DDColor 已有暖调的图上
#   感知不强），v2 上调约 1.6~2 倍并加 l_drop，目标"点选即一眼可辨"。
_PRESETS = {
    "warm":      {"name": "暖调", "da": +11.0, "db": +8.0, "sat_k": 1.08,
                  "l_gain": 0.00,  "l_bias": 0.000, "l_drop": 0.06},
    "cold":      {"name": "冷调", "da": -6.0, "db": -11.0, "sat_k": 1.00,
                  "l_gain": 0.00,  "l_bias": 0.000, "l_drop": 0.04},
    "nostalgic": {"name": "怀旧", "da": +4.0, "db": +14.0, "sat_k": 0.72,
                  "l_gain": -0.10, "l_bias": 0.03, "l_drop": 0.10},
    "vivid":     {"name": "鲜艳", "da": 0.0,  "db": 0.0,  "sat_k": 1.38,
                  "l_gain": 0.00,  "l_bias": 0.000, "l_drop": 0.00},
    # 淡雅 v2.1：与怀旧（暖黄泛旧）拉开方向——微冷 + 低饱和 + 提亮，
    # 呈"粉白素雅"，两者不再读感相近（v2 目检发现近亲问题当日微调）
    "soft":      {"name": "淡雅", "da": -2.0, "db": -3.0, "sat_k": 0.60,
                  "l_gain": 0.00,  "l_bias": 0.050, "l_drop": 0.00},
    # "保持黑白"＝跳过智能上色（pipeline 侧不再加载 DDColor），非色调参数
    "bw":        {"name": "保持黑白", "bw": True},
}

# 关键词词典：chip 文案即键，另收录常见近义说法（自由文本也按子串匹配）。
# 扫描顺序即优先级：bw 语义最强（跳过上色），其余按可辨识度排序。
_SCAN = ["bw", "nostalgic", "warm", "cold", "vivid", "soft"]
_KEYWORDS = {
    "bw":        ("保持黑白", "黑白", "不上色", "黑白保留", "黑白照片"),
    "nostalgic": ("怀旧", "泛黄", "年代感", "复古", "老照片感"),
    "warm":      ("暖调", "暖色", "暖色调", "偏暖", "暖一点", "暖色系"),
    "cold":      ("冷调", "冷色", "冷色调", "偏冷", "冷一点", "冷色系"),
    "vivid":     ("鲜艳", "浓艳", "艳丽", "色彩浓郁", "鲜明"),
    "soft":      ("淡雅", "素雅", "柔和", "清淡", "低饱和"),
}

# ---- 安全钳制（预设已在区间内；防未来改预设越界毁图） ----
# v2：区间随预设幅度上调同步放宽，仍属"滤镜级"而非"毁图级"（对比商业
# 滤镜 LUT 的常规强度），未识别直通 + 空偏好零改动护栏不变。
MAX_AB_SHIFT = 15.0          # |Δa|/|Δb| 上限
SAT_LO, SAT_HI = 0.55, 1.45  # 饱和缩放区间
MAX_L_GAIN = 0.18            # 对比度调整幅度上限
MAX_L_BIAS = 0.05            # 明度偏移上限（归一化）
MAX_L_DROP = 0.12            # 高光压亮幅度上限（归一化）
FACE_RELIEF = 0.5            # 人脸框内色温偏移强度系数（0.5=减半）


def _clamp_params(p: dict) -> dict:
    """预设参数钳制到安全区间（防越界；L45 护栏之一）。"""
    return {
        "da": float(np.clip(p.get("da", 0.0), -MAX_AB_SHIFT, MAX_AB_SHIFT)),
        "db": float(np.clip(p.get("db", 0.0), -MAX_AB_SHIFT, MAX_AB_SHIFT)),
        "sat_k": float(np.clip(p.get("sat_k", 1.0), SAT_LO, SAT_HI)),
        "l_gain": float(np.clip(p.get("l_gain", 0.0), -MAX_L_GAIN, MAX_L_GAIN)),
        "l_bias": float(np.clip(p.get("l_bias", 0.0), -MAX_L_BIAS, MAX_L_BIAS)),
        "l_drop": float(np.clip(p.get("l_drop", 0.0), 0.0, MAX_L_DROP)),
    }


def parse_prompt(text: str) -> dict | None:
    """自由文本 → 预设参数 dict（含 name/key）；未命中任何偏好返回 None。

    pipeline 拿到 None 即直通（空提示词零改动护栏）。bw 预设带 "bw": True，
    由 pipeline 侧解释为"跳过智能上色"。
    """
    t = (text or "").strip()
    if not t:
        return None
    # 否定 bw 误伤防护："不要黑白照片，就想要彩色的" 不应命中 bw
    neg_bw = any(n in t for n in ("不要黑白", "别黑白", "不要保持黑白", "不是黑白"))
    for key in _SCAN:
        if key == "bw" and neg_bw:
            continue
        for kw in _KEYWORDS[key]:
            if kw in t:
                p = dict(_PRESETS[key])
                p["key"] = key
                return p
    return None


# ---- 人脸保护（Haar 框 + 羽化软权重，模式同 deglare.py；纯 CPU） ----
_FACE_CASC = None


def _face_casc():
    global _FACE_CASC
    if _FACE_CASC is None:
        _FACE_CASC = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE_CASC


def _face_weight(bgr: np.ndarray):
    """返回 (软权重图 0~1, 检出人脸数)。框内 1.0、框缘高斯羽化到 0。
    检测失败/无脸 → 全零权重（全图统一偏移，不影响功能）。"""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fc = _face_casc()
    faces = fc.detectMultiScale(gray, 1.05, 6, minSize=(80, 80))
    if not len(faces):
        faces = fc.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    mask = np.zeros((h, w), dtype=np.float32)
    if not len(faces):
        return mask, 0
    for (fx, fy, fw, fh) in faces:
        mask[int(fy):int(fy + fh), int(fx):int(fx + fw)] = 1.0
    # 羽化半径随图尺寸走（大图框缘硬切会留可见边界带）
    k = int(max(9, min(h, w) // 60))
    k += (k + 1) % 2   # 取奇数
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask, len(faces)


def apply_tone(bgr: np.ndarray, preset: dict):
    """按预设做 Lab 色调调整，返回 (调整后 uint8 BGR, 日志消息)。

    顺序：先饱和缩放（色相不变）→ 再色温偏移（人脸框内减半）→ 最后 L 对比/
    明度/高光压亮（l_drop 仅对 L>0.55 亮部平滑生效，使过曝白区可显色调）。
    bw 预设不进入本函数（pipeline 侧已跳过上色），防御性直通。
    """
    if preset is None or preset.get("bw"):
        return bgr, ""
    p = _clamp_params(preset)
    da, db, sat_k = p["da"], p["db"], p["sat_k"]
    l_gain, l_bias, l_drop = p["l_gain"], p["l_bias"], p["l_drop"]

    face_w, n_faces = _face_weight(bgr)
    relief = 1.0 - FACE_RELIEF * face_w   # 脸内 0.5 → 框外 1.0

    img01 = (bgr.astype(np.float32)) / 255.0
    lab = cv2.cvtColor(img01, cv2.COLOR_BGR2LAB)   # float32: L∈[0,100], ab∈[-127,127]
    L = lab[:, :, 0] / 100.0
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    a = a * sat_k + da * relief
    b = b * sat_k + db * relief
    L = (L - 0.5) * (1.0 + l_gain) + 0.5 + l_bias
    if l_drop > 0:
        # 高光压亮（v2）：Lab 白点 L=100 处 ab 恒 0、转 BGR 必裁剪回白，
        # 过曝白区须降 L 后 ab 偏移才可感。仅压 L>0.55 的亮部（平滑权重）
        hl = np.clip((L - 0.55) / 0.45, 0.0, 1.0)
        L = L - l_drop * hl
    lab = np.dstack([np.clip(L, 0.0, 1.0) * 100.0,
                     np.clip(a, -127.0, 127.0),
                     np.clip(b, -127.0, 127.0)]).astype(np.float32)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = np.clip(out * 255.0, 0, 255).round().astype(np.uint8)

    parts = [preset.get("name", preset.get("key", "?"))]
    if da or db:
        parts.append(f"Δa{da:+.0f}/Δb{db:+.0f}")
    if sat_k != 1.0:
        parts.append(f"饱和×{sat_k:.2f}")
    if l_gain:
        parts.append(f"对比{'' if l_gain > 0 else '−'}{abs(l_gain) * 100:.0f}%")
    if l_bias:
        parts.append(f"明度{'' if l_bias > 0 else '−'}{abs(l_bias) * 100:.0f}%")
    if l_drop:
        parts.append(f"高光压{'' if l_drop > 0 else '−'}{abs(l_drop) * 100:.0f}%")
    parts.append(f"人脸保护 {n_faces} 框" if n_faces else "无人脸")
    return out, "／".join(parts)


def _main():
    ap = argparse.ArgumentParser(description="L45 修复偏好提示词单测入口")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    preset = parse_prompt(args.prompt)
    if preset is None:
        print("[tone] 未命中偏好关键词，直通（返回 None）")
        return
    print(f"[tone] 命中偏好：{preset['name']}（key={preset['key']}）")
    if preset.get("bw"):
        print("[tone] 保持黑白：pipeline 侧将跳过智能上色（本工具无输出）")
        return
    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"[!] 无法读取：{args.input}")
    out, msg = apply_tone(img, preset)
    print(f"[tone] {msg}")
    if args.out:
        cv2.imwrite(args.out, out)
        print(f"[tone] 已写出：{args.out}")


if __name__ == "__main__":
    _main()
