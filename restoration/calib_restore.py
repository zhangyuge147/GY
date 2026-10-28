# -*- coding: utf-8 -*-
"""
任务 3.9：考据校色辅助工具（calib_restore.py）
================================================
修复流水线六步：去网纹 → 划痕补全 → 超分辨率 → 人脸精修 → 智能上色 → 考据校色
本模块实现最后一步"考据校色"的人工辅助：人工对照色彩考据库做局部色值修正。

为什么是"人工校色"：
  DDColor（3.7）只负责"自然着色"，色值不一定符合史实（局限 L20）；
  老照片为灰度，色相信息丢失，任何模型都无法从灰度图推理出史实色（L25）。
  故项目已定"人工对照考据表校色"方案（不采用模型自动识别）。
  本工具为人工校色提供辅助：加载考据色表 → 参考色板 → 局部区域 LAB 色偏修正
  → 前后对比 + 改动记录。

实现说明：
  1. COLOR_REFERENCE 为结构化考据色表（RGB，供内部校色对照），与素材库
     04-色彩考据参考/*.txt 考据文档对应；文档为权威来源，此表为其色值摘要。
  2. 核心操作 apply_color_cast：把指定 mask 区域内像素的 LAB 色相/饱和度
     向目标考据色拉近（强度 strength 可调），亮度 L 通道保持不变，
     避免破坏光影结构。
  3. 输出校色图 + 色板参考图 + 前后并排对比图 + 校色记录 txt（含 ΔE）。

用法（guangying 环境下，在 guangying/ 目录运行）：
  python restoration/calib_restore.py --list                          # 列出考据色表
  python restoration/calib_restore.py --image xx.png --person 杨靖宇 --target 军上衣 \
        --mask mask.png --strength 0.5 --out xx_calib.png             # 单张校色调试
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent          # restoration/

# ------------------------------------------------------------------
# 结构化考据色表（RGB 0-255）
# 来源：素材库/04-色彩考据参考/ 下考据文档（2026-10-17 网络调研撰写）。
# 色值为近似参考，标注"待实物复核"；档位化约束，校色时按部位选档。
# ------------------------------------------------------------------
COLOR_REFERENCE = {
    "党旗": {   # 旗帜标识/党旗标准色.txt
        "旗面":   (230, 0, 0),      # 标准中国红
        "党徽":   (253, 185, 19),   # 金黄色
    },
    "八路军": {  # 军服徽章/八路军军服.txt
        "军服前期灰": (150, 150, 150),   # 八爷灰
        "军服后期土黄": (166, 124, 82),
        "军服后期草绿": (120, 140, 90),
        "红五星帽徽": (200, 20, 30),
        "红领章":   (200, 20, 30),
        "绑腿":     (140, 145, 150),
    },
    "杨靖宇": {  # 军服徽章/杨靖宇东北抗联军服.txt
        "军上衣":   (166, 124, 82),   # 黄军装（黄菠萝树皮煮染）
        "军大衣":   (175, 140, 95),   # 冬季黄棉大衣
        "战斗帽":   (160, 120, 78),
        "红疙瘩":   (200, 30, 40),    # 五角星红疙瘩
        "红领章":   (200, 30, 40),
        "绑腿":     (120, 140, 90),
        "胶鞋":     (40, 40, 45),
    },
    "吉鸿昌": {  # 军服徽章/吉鸿昌西北军同盟军服.txt
        "军服灰布": (135, 138, 145),   # 西北军灰布
        "武装带":   (110, 85, 60),
        "斗篷":     (75, 75, 80),      # 就义照
        "礼帽":     (60, 60, 65),
    },
    "彭雪枫": {  # 军服徽章/彭雪枫新四军军服.txt
        "军服灰":   (135, 140, 150),   # 新四军灰
        "军服蓝灰": (125, 132, 148),
        "N4A臂章底": (225, 225, 220),
        "N4A臂章字": (50, 70, 110),
        "武装带":   (110, 85, 60),
    },
    "焦裕禄": {  # 干部服饰/焦裕禄干部服.txt
        "中山装新": (70, 75, 90),      # 深灰蓝干部装
        "中山装旧": (110, 115, 125),   # 洗旧发灰
        "白秋衣":   (225, 225, 215),
        "毛背心":   (166, 124, 82),    # 土黄色鸡心领
        "补丁":     (60, 65, 80),
    },
    "杨贵": {    # 干部服饰/杨贵干部服.txt
        "中山装新": (75, 80, 95),
        "中山装旧": (115, 120, 130),
        "草帽":     (185, 160, 110),
        "布鞋":     (60, 55, 50),
    },
    "红旗渠场景": {  # 场景参考/wiki_红旗渠*.jpg
        "石壁":   (150, 150, 155),   # 太行青灰石壁
        "渠水":   (90, 140, 130),    # 清绿山泉
    },
    "兰考场景": {  # 场景参考/wiki_兰考纪念园_01.jpg
        "黄土":   (180, 150, 100),
        "泡桐绿": (90, 120, 70),
        "泡桐树干": (120, 100, 85),
    },
}

DEFAULT_STRENGTH = 0.5   # 色偏修正强度 0-1（0=不动，1=完全拉向目标色）


def rgb2bgr(c):
    return (c[2], c[1], c[0])


def palette_bgr(person: str) -> np.ndarray:
    """生成参考色板（横条，每行一个部位名+色块）——人工校色对照用"""
    items = list(COLOR_REFERENCE[person].items())
    cell_w, cell_h = 120, 36
    pad = 8
    w = cell_w * 3 + pad
    h = len(items) * (cell_h + pad) + pad
    board = np.full((h, w, 3), 245, np.uint8)
    for i, (name, rgb) in enumerate(items):
        y = pad + i * (cell_h + pad)
        bgr = rgb2bgr(rgb)
        cv2.rectangle(board, (pad, y), (pad + cell_w, y + cell_h), bgr, -1)
        cv2.rectangle(board, (pad, y), (pad + cell_w, y + cell_h), (0, 0, 0), 1)
        cv2.putText(board, f"{name}  RGB{rgb}",
                    (pad + cell_w + 8, y + cell_h // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    return board


def apply_color_cast(bgr: np.ndarray, mask: np.ndarray, target_rgb, strength=DEFAULT_STRENGTH):
    """局部 LAB 色偏修正：mask 内像素的色相/饱和度向 target_rgb 拉近，L 亮度不变。

    bgr:     uint8 BGR 彩色图（DDColor 上色结果，即 05_color.png）
    mask:    uint8 单通道，非零区域 = 待校色区域（人工圈选）
    target_rgb: 考据色 RGB
    strength:   0-1，0=不动，1=完全替换为目标的 a/b 偏置
    返回:    uint8 BGR（mask 外像素零改动）
    """
    if bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(f"图与 mask 尺寸不一致：{bgr.shape[:2]} vs {mask.shape[:2]}")
    m = (mask > 0)
    if m.sum() == 0:
        return bgr.copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    t = cv2.cvtColor(np.uint8([[rgb2bgr(target_rgb)]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    # 区域当前平均 a/b
    cur_a = lab[m, 1].mean()
    cur_b = lab[m, 2].mean()
    # a/b 向目标偏移 strength 比例；L 保持不变
    lab[m, 1] += (t[1] - cur_a) * strength
    lab[m, 2] += (t[2] - cur_b) * strength
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def delta_e(bgr_a: np.ndarray, bgr_b: np.ndarray, mask: np.ndarray):
    """mask 区域平均色差（Lab 欧氏距离近似 ΔE76）"""
    m = mask > 0
    la = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2LAB).astype(np.float32)[m]
    lb = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2LAB).astype(np.float32)[m]
    return float(np.sqrt(((la - lb) ** 2).sum(axis=1)).mean())


class Calibrator:
    """考据校色器：加载人物考据色表，提供人工校色辅助操作"""

    def __init__(self, person: str, strength: float = DEFAULT_STRENGTH):
        if person not in COLOR_REFERENCE:
            raise SystemExit(
                f"[!] 未找到人物考据色表：{person}\n"
                f"    可用人物：{', '.join(COLOR_REFERENCE.keys())}")
        self.person = person
        self.strength = float(strength)
        self.colors = COLOR_REFERENCE[person]

    def calibrate(self, bgr: np.ndarray, mask: np.ndarray, target: str,
                  strength: float = None) -> np.ndarray:
        if target not in self.colors:
            raise ValueError(f"[!] 部位 '{target}' 不在 {self.person} 考据色表中。"
                             f"可选：{', '.join(self.colors)}")
        s = self.strength if strength is None else float(strength)
        return apply_color_cast(bgr, mask, self.colors[target], s)

    def save_report(self, before, after, mask, target, strength, out_txt: Path):
        """校色记录：人物/部位/强度/区域 ΔE/时间——供人工复核留档"""
        d = delta_e(before, after, mask)
        report = (f"考据校色记录（{self.person} · {target} · 强度 {strength}）\n"
                  f"========================================\n"
                  f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"人物考据表：{self.person}\n"
                  f"校色部位：{target} 目标色 RGB{self.colors[target]}\n"
                  f"修正强度：{strength}\n"
                  f"校色区域：mask 覆盖 {(mask > 0).mean() * 100:.2f}% 像素\n"
                  f"区域平均色差 ΔE：{d:.2f}\n"
                  f"说明：亮度 L 保持不变，仅 a/b 色相饱和度向考据色拉近；\n"
                  f"      色值来源素材库 04-色彩考据参考 考据文档（近似参考，待实物复核）。\n")
        out_txt.write_text(report, encoding="utf-8")
        return report


def main():
    ap = argparse.ArgumentParser(
        description="光影重生·考据校色辅助（任务3.9）：人工对照考据色表做局部 LAB 色偏修正")
    ap.add_argument("--list", action="store_true", help="列出考据色表并退出")
    ap.add_argument("--image", type=str, help="输入彩色图路径（DDColor 上色结果）")
    ap.add_argument("--person", type=str, help="人物/考据表名")
    ap.add_argument("--target", type=str, help="校色部位（如 军上衣/红领章/中山装旧）")
    ap.add_argument("--mask", type=str, help="校色区域 mask（非零=待校色）")
    ap.add_argument("--strength", type=float, default=DEFAULT_STRENGTH,
                    help=f"色偏修正强度 0-1（默认 {DEFAULT_STRENGTH}）")
    ap.add_argument("--out", type=str, default=None, help="输出前缀（生成 _calib.png/对比/记录）")
    args = ap.parse_args()

    if args.list:
        for p, items in COLOR_REFERENCE.items():
            print(f"{p}: " + "; ".join(f"{n}=RGB{c}" for n, c in items.items()))
        return

    if not (args.image and args.person and args.target and args.mask):
        ap.error("请指定 --image --person --target --mask（或 --list）")

    from preprocess import imread_bgr, imwrite_unicode
    img = imread_bgr(Path(args.image))
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"[!] 无法读取 mask：{args.mask}")

    cal = Calibrator(args.person, args.strength)
    t0 = time.time()
    result = cal.calibrate(img, mask, args.target)
    dt = time.time() - t0

    prefix = Path(args.out) if args.out else Path(args.image).with_suffix("")
    cal_png = Path(str(prefix) + "_calib.png")
    imwrite_unicode(cal_png, result)

    # 前后并排对比图
    cmp = np.hstack([cv2.resize(img, (img.shape[1], img.shape[0])), result])
    cmp_png = Path(str(prefix) + "_对比.png")
    imwrite_unicode(cmp_png, cmp)

    # 参考色板
    board_png = Path(str(prefix) + "_色板.png")
    imwrite_unicode(board_png, palette_bgr(args.person))

    report = cal.save_report(img, result, mask, args.target, args.strength,
                             Path(str(prefix) + "_校色记录.txt"))
    print(report)
    print(f"[OK] 校色完成 {dt:.2f}s -> {cal_png.name}（对比 {cmp_png.name}，色板 {board_png.name}）")


if __name__ == "__main__":
    main()
