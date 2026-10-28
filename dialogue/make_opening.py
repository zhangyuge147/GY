# -*- coding: utf-8 -*-
"""
开端照片标题图生成器（阶段5 · 5.3）
用途：程序化生成「光影重生」红色主题开端照片，作为视频片头静帧。

设计要点：
  - 纯 Pillow + numpy 程序化绘制，零外部素材、零版权风险、可全自动复现
  - 红色渐变底 + 山峦/长城剪影 + 硝烟烟雾 + 放射光束（"光影"意象）
  - 金色渐变标题 + 五角星 + 两侧麦穗，主字体楷体

用法：
  python make_opening.py [输出路径] [--size W H] [--title 标题] [--subtitle 副标题]
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------- 配色
# 红色主题（深红 -> 暗红）
RED_TOP = (0x8B, 0x1A, 0x1A)      # 上
RED_BOTTOM = (0x3A, 0x00, 0x00)   # 下

# 金色（标题/五星/麦穗）
GOLD_TOP = (0xFF, 0xE0, 0x8A)     # 亮金
GOLD_BOTTOM = (0xD4, 0xAF, 0x37)  # 暗金

# 剪影色（由远到近，越近越暗，形成明显层次）
SHADOW_MOUNTAIN_FAR = (0x6E, 0x20, 0x20)
SHADOW_MOUNTAIN_NEAR = (0x50, 0x14, 0x14)
SHADOW_WALL = (0x14, 0x03, 0x03)
WALL_TOWER = (0x2E, 0x0A, 0x0A)  # 烽火台主体略亮，与城墙带拉开层次

# 硝烟主色（偏亮灰红，与深红底拉开对比）
SMOKE_COLOR = (0xB2, 0x78, 0x62)

# ---------------------------------------------------------------- 字体
FONT_CANDIDATES = {
    "kai": [r"C:\Windows\Fonts\simkai.ttf",
            "/usr/share/fonts/truetype/arphic/ukai.ttc"],
    "hei": [r"C:\Windows\Fonts\simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"],
    "yahei_bold": [r"C:\Windows\Fonts\msyhbd.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"],
}


def load_font(kind, size):
    """按种类加载中文字体，跨平台兜底，找不到则抛错。"""
    for path in FONT_CANDIDATES[kind]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise RuntimeError(f"未找到中文字体（kind={kind}），请安装 SimKai/SimHei/微软雅黑")


# ---------------------------------------------------------------- 基础绘制
def gradient_bg(w, h, top, bottom):
    """垂直渐变背景，返回 RGB。"""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.array(top, dtype=np.float32)
    bottom = np.array(bottom, dtype=np.float32)
    arr = (top * (1.0 - t) + bottom * t).astype(np.uint8)
    arr = np.repeat(arr, w, axis=1)  # (h,1,3) -> (h,w,3)
    return Image.fromarray(arr, "RGB")


def radial_glow(size, center, radius, color, max_alpha=90):
    """径向渐变光晕（alpha 随距离线性衰减），返回 RGBA。"""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = center
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(1.0 - dist / radius, 0.0, 1.0) * max_alpha
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 0:3] = np.array(color, dtype=np.uint8)
    layer[..., 3] = alpha.astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def draw_text_gold(img, text, font, center, shadow=True):
    """在 RGBA 图上绘制带金色渐变与阴影的文字。"""
    w, h = img.size
    cx, cy = center
    # 1) 文字 mask
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((cx, cy), text, font=font, fill=255, anchor="mm")

    # 2) 阴影（偏移 + 半透明深色）
    if shadow:
        smask = ImageChops.offset(mask, 5, 7)
        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        shadow_layer.paste((20, 0, 0, 150), (0, 0), smask)
        img = Image.alpha_composite(img, shadow_layer)

    # 3) 金色渐变
    gold = gradient_bg(w, h, GOLD_TOP, GOLD_BOTTOM)
    img.paste(gold, (0, 0), mask)
    return img


# ---------------------------------------------------------------- 场景元素
def draw_mountain(draw, w, baseline, amp, phase, color, layers=1):
    """山峦剪影：一条沿 x 波浪起伏的山脊线，向下填充到画面底部。"""
    n = 160
    xs = np.linspace(0, w, n)
    ys = baseline + amp * (
        np.sin(xs / 170.0 + phase)
        + 0.5 * np.cos(xs / 95.0 + phase * 2)
        + 0.3 * np.sin(xs / 43.0 + phase * 3)
    )
    # 归一化到 [0, 1] 避免波形过平
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-6)
    ys = baseline + amp * ys
    pts = list(zip(xs, ys))
    poly = pts + [(w, baseline + amp * 2), (0, baseline + amp * 2)]
    draw.polygon(poly, fill=color)


def draw_great_wall(draw, w, top_y, wall_h, color):
    """长城剪影：横向城墙带 + 顶部垛口。"""
    # 城墙基座
    draw.rectangle([0, top_y, w, top_y + wall_h], fill=color)
    # 垛口
    crenel_w, crenel_h, gap = 16, 18, 22
    x = 0
    while x < w:
        draw.rectangle([x, top_y - crenel_h, x + crenel_w, top_y], fill=color)
        x += crenel_w + gap


def draw_watchtower(draw, cx, top_y, tower_w, tower_h, color):
    """烽火台剪影：墙面上方凸起的塔楼，顶部带垛口与箭窗。"""
    x0, x1 = int(cx - tower_w // 2), int(cx + tower_w // 2)
    y0 = int(top_y - tower_h)
    draw.rectangle([x0, y0, x1, top_y], fill=color)
    # 塔顶垛口
    cw, ch, gap = 12, 14, 16
    x = x0
    while x < x1:
        draw.rectangle([x, y0 - ch, x + cw, y0], fill=color)
        x += cw + gap
    # 箭窗（更暗，增强可读性）
    win_w = int(tower_w * 0.16)
    win_x0 = cx - win_w // 2
    win_y0 = int(y0 + tower_h * 0.32)
    win_y1 = int(y0 + tower_h * 0.46)
    win_color = (max(color[0] - 14, 0), max(color[1] - 3, 0), max(color[2] - 3, 0))
    draw.rectangle([win_x0, win_y0, cx + win_w // 2, win_y1], fill=win_color)
    # 塔顶轮廓光（受光面，增强可读性与层次）
    draw.line([x0, y0, x1, y0], fill=(0xC0, 0x90, 0x40, 220), width=3)
    draw.line([x0, y0, x0, top_y], fill=(0xB0, 0x80, 0x38, 140), width=2)
    draw.line([x1, y0, x1, top_y], fill=(0xB0, 0x80, 0x38, 140), width=2)


def draw_beacon_smoke(draw, cx, top_y, seed):
    """在烽火台塔顶绘制上升扩散的烽烟团（软椭圆，随后统一高斯模糊）。"""
    rng = np.random.default_rng(seed)
    for dy, r, alpha in ((24, 30, 230), (62, 52, 190),
                         (112, 82, 150), (175, 120, 110)):
        ex = cx + rng.uniform(-20, 20)
        ey = top_y - dy
        draw.ellipse([ex - r, ey - r, ex + r, ey + r],
                     fill=(0xEC, 0xE6, 0xD8, alpha))


def star_points(cx, cy, r_outer, r_inner, n=5, rotation=-90.0):
    """五角星顶点（外顶点 + 内顶点）。"""
    pts = []
    for i in range(n * 2):
        r = r_outer if i % 2 == 0 else r_inner
        ang = math.radians(rotation + i * (360.0 / (n * 2)))
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def draw_wheat(draw, base, height, lean, color):
    """麦穗：弧形茎 + 顶部麦芒 + 交替麦粒（椭圆）。"""
    bx, by = base
    tx, ty = bx + lean, by - height
    # 茎
    draw.line([bx, by, tx, ty], fill=color, width=6)
    # 顶部麦芒
    for a in (-0.55, 0.0, 0.55):
        draw.line([tx, ty, tx + lean * 0.35 + a * 42, ty - 34], fill=color, width=2)
    # 麦粒：沿茎交替两侧椭圆
    steps = 10
    for i in range(1, steps):
        t = i / (steps - 1)
        px, py = bx + lean * t, by - height * t
        for s in (-1.0, 1.0):
            ox = px + s * 13
            oy = py - 9
            draw.ellipse([ox - 8, oy - 11, ox + 8, oy + 11], fill=color)


def draw_beams(draw, center, radius, color, count=14, alpha=55):
    """从中心放射的光束（细长三角，整体低 alpha 金色）。"""
    cx, cy = center
    r0 = radius * 0.18
    for i in range(count):
        ang = math.radians(i * (360.0 / count))
        dx, dy = math.cos(ang), math.sin(ang)
        # 光束两侧端点
        ex, ey = cx + dx * radius, cy + dy * radius
        px, py = -dy, dx  # 垂直方向
        half = radius * 0.06
        poly = [
            (cx + px * r0 * 0.2, cy + py * r0 * 0.2),
            (cx - px * r0 * 0.2, cy - py * r0 * 0.2),
            (ex - px * half, ey - py * half),
            (ex + px * half, ey + py * half),
        ]
        draw.polygon(poly, fill=(*color, alpha))


# ---------------------------------------------------------------- 主流程
def make_opening(out_path, size=(1920, 1080), title="光影重生",
                 subtitle="红色历史影像修复与数字人交互"):
    w, h = size
    # 1) 渐变背景
    img = gradient_bg(w, h, RED_TOP, RED_BOTTOM).convert("RGBA")

    # 2) 中心金色光晕（聚焦标题区）
    glow = radial_glow(size, (w // 2, int(h * 0.32)), w * 0.55,
                       (0xFF, 0xD0, 0x70), max_alpha=60)
    img = Image.alpha_composite(img, glow)
    draw = ImageDraw.Draw(img, "RGBA")  # composite 返回新对象，须重建 draw

    # 3) 山峦剪影（远景 + 近景两层）
    draw_mountain(draw, w, int(h * 0.72), 120, 1.3, SHADOW_MOUNTAIN_FAR)
    draw_mountain(draw, w, int(h * 0.78), 90, 4.1, SHADOW_MOUNTAIN_NEAR)

    # 4) 长城剪影（城墙 + 烽火台）
    wall_top = int(h * 0.80)
    draw_great_wall(draw, w, wall_top, int(h * 0.08), SHADOW_WALL)
    for cx in (int(w * 0.18), int(w * 0.50), int(w * 0.82)):
        draw_watchtower(draw, cx, wall_top, 150, int(h * 0.16), WALL_TOWER)

    # 5) 烽烟（每座烽火台塔顶冒烟，软烟团上升扩散）
    smoke_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    smoke_draw = ImageDraw.Draw(smoke_layer, "RGBA")
    for i, cx in enumerate((int(w * 0.18), int(w * 0.50), int(w * 0.82))):
        draw_beacon_smoke(smoke_draw, cx, wall_top - int(h * 0.16), seed=11 + i)
    img = Image.alpha_composite(img, smoke_layer.filter(ImageFilter.GaussianBlur(10)))
    draw = ImageDraw.Draw(img, "RGBA")

    # 6) 放射光束 + 五角星
    center = (w // 2, int(h * 0.30))
    beams = Image.new("RGBA", size, (0, 0, 0, 0))
    draw_beams(ImageDraw.Draw(beams), center, w * 0.5, (0xFF, 0xD0, 0x70),
               count=14, alpha=45)
    img = Image.alpha_composite(img, beams.filter(ImageFilter.GaussianBlur(4)))
    draw = ImageDraw.Draw(img, "RGBA")
    # 五角星
    star = star_points(center[0], center[1], 72, 29)
    draw.polygon(star, fill=(*GOLD_TOP, 255), outline=(*GOLD_BOTTOM, 255))

    # 7) 麦穗（左右各一束，置于标题两侧）
    draw_wheat(draw, (int(w * 0.15), int(h * 0.86)), int(h * 0.30), -70, (*GOLD_TOP, 255))
    draw_wheat(draw, (int(w * 0.85), int(h * 0.86)), int(h * 0.30), 70, (*GOLD_TOP, 255))

    # 8) 主标题（楷体，金色渐变 + 阴影）
    title_font = load_font("kai", int(h * 0.17))
    img = draw_text_gold(img, title, title_font, (w // 2, int(h * 0.42)), shadow=True)

    # 9) 副标题
    sub_font = load_font("hei", int(h * 0.045))
    img = draw_text_gold(img, subtitle, sub_font, (w // 2, int(h * 0.56)), shadow=True)

    # 10) 四角小字角标
    corner = "光影重生 · 红色影像修复"
    corner_font = load_font("hei", int(h * 0.02))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((30, 20), corner, font=corner_font, fill=(0xE0, 0xD0, 0xA0, 200))
    draw.text((w - 30, h - 50), corner, font=corner_font, fill=(0xE0, 0xD0, 0xA0, 200),
              anchor="rs")

    img.convert("RGB").save(out_path, quality=95)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="生成「光影重生」开端照片")
    ap.add_argument("out", nargs="?", default="opening.png", help="输出 PNG 路径")
    ap.add_argument("--size", nargs=2, type=int, default=[1920, 1080], metavar=("W", "H"))
    ap.add_argument("--title", default="光影重生")
    ap.add_argument("--subtitle", default="红色历史影像修复与数字人交互")
    args = ap.parse_args()
    path = make_opening(args.out, tuple(args.size), args.title, args.subtitle)
    print(f"已生成开端照片：{os.path.abspath(path)}")


if __name__ == "__main__":
    main()