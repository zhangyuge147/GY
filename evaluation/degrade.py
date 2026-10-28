# -*- coding: utf-8 -*-
"""
光影重生 · 合成退化基准集生成器（任务 3.3）
==========================================
对 restoration/data/bench/ 的 20 张现代高清图人工"做旧"，生成合成退化
基准集，供修复流水线（任务 3.10）跑全流程后与原图对比计算 PSNR/SSIM/LPIPS。

为什么需要合成退化集：
  真实老照片没有"干净原图"，无法计算有参考指标；把现代高清图人为退化成
  "老照片"，就有了 (退化图, 原图) 配对，参考指标才有意义。原始 bench 图
  本身就是评测的参考答案（ground truth），本脚本只读不改。

退化顺序（模拟一张老照片的形成过程）：
  彩色原图
    → ① 转黑白      老照片本身是黑白的（DDColor 上色环节的评测对象）
    → ② 缩小分辨率  模拟老照片扫描件分辨率（默认长边 1200px；同时给
                     超分留提升空间、控制后续评测计算量）
    → ③ 高斯模糊    模拟老镜头/印刷/翻拍的综合锐度损失
    → ④ 划痕        物理损伤：随机游走折线，白色相纸划痕为主、少量深色
                     擦痕，条数/宽度/透明度随机（LaMa 补全的评测对象）
    → ⑤ 噪点        高斯噪声（胶片颗粒）+ 椒盐噪声（灰尘颗粒）
  输出 PNG 无损保存，避免 JPEG 压缩伪影污染指标。

可复现性：每张图随机种子 = crc32(文件名) + --seed 偏移，重跑结果一致；
逐张实际参数记录在 evaluation/data/degraded_params.csv（测试说明文档
任务 7.7 的直接材料）。

用法（在 guangying/ 目录、guangying 环境下）：
  python evaluation/degrade.py                        # bench 全部图片
  python evaluation/degrade.py --long-edge 1000       # 退化图长边改 1000
  python evaluation/degrade.py --seed 1 --out data/degraded_v2   # 换一批随机参数
"""
import argparse
import csv
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent            # evaluation/
PROJECT = ROOT.parent                             # guangying/
DEFAULT_BENCH = PROJECT / "restoration" / "data" / "bench"
DEFAULT_OUT = ROOT / "data" / "degraded"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---- 退化参数取值范围（"做旧"力度，可按 3.10 评测反馈再调）----
BLUR_KERNELS = [3, 5, 7, 9]           # 高斯模糊核候选（奇数）
BLUR_SIGMA = (0.8, 2.2)               # 高斯模糊 sigma
SCRATCH_COUNT = (3, 8)                # 每图划痕条数
NOISE_SIGMA = (5, 12)                 # 高斯噪声 sigma（胶片颗粒感）
SP_DENSITY = (0.002, 0.008)           # 椒盐噪声密度（灰尘颗粒）


# ---------------------------------------------------------------- 基础 IO
def imread_bgr(path: Path) -> np.ndarray:
    """读图：兼容中文路径与 EXIF 方向，返回 BGR 三通道"""
    img = ImageOps.exif_transpose(Image.open(path))
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def imwrite_unicode(path: Path, img: np.ndarray):
    """写图：兼容中文路径"""
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"图片编码失败：{path}")
    buf.tofile(path)


# ---------------------------------------------------------------- 退化算子
def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    """② 缩小到指定长边（只缩不放），INTER_AREA 抗锯齿"""
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def add_scratches(img: np.ndarray, rng) -> tuple:
    """④ 划痕：随机游走折线，白色为主（相纸划痕），约 1/4 为深色擦痕"""
    h, w = img.shape[:2]
    n = int(rng.integers(*SCRATCH_COUNT))
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        theta = rng.uniform(0, 2 * np.pi)
        pts = [(x, y)]
        for _ in range(int(rng.integers(15, 50))):     # 随机游走 15~50 段
            theta += rng.uniform(-0.45, 0.45)
            step = rng.uniform(6, 18)
            x += step * np.cos(theta)
            y += step * np.sin(theta)
            pts.append((x, y))
        pts = np.round(np.array(pts)).astype(np.int32)
        dark = rng.random() < 0.25
        color = int(rng.integers(0, 60)) if dark else int(rng.integers(190, 256))
        thick = int(rng.integers(1, 4))
        alpha = rng.uniform(0.55, 0.95)
        layer = img.copy()
        cv2.polylines(layer, [pts], False, (color, color, color), thick)
        img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    return img, n


def add_noise(img: np.ndarray, rng) -> tuple:
    """⑤ 噪点：高斯噪声（颗粒）+ 椒盐噪声（灰尘）。
    高斯噪声按单通道采样再广播到三通道，保证灰度图不引入彩色噪点"""
    sigma = rng.uniform(*NOISE_SIGMA)
    density = rng.uniform(*SP_DENSITY)
    img = img.astype(np.float32) + rng.normal(0, sigma, (*img.shape[:2], 1))
    img = np.clip(img, 0, 255)
    m = rng.random(img.shape[:2])                       # 椒盐用同一随机面
    img[m < density / 2] = 255
    img[(m >= density / 2) & (m < density)] = 0
    return img.astype(np.uint8), sigma, density


def degrade(img: np.ndarray, long_edge: int, rng: np.ndarray) -> tuple:
    """单张退化主流程：转黑白 → 缩小 → 模糊 → 划痕 → 噪点，返回 (退化图, 参数dict)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)        # 保持 3 通道，流水线统一按彩色处理
    out = resize_long_edge(out, long_edge)               # ②
    ksize = int(rng.choice(BLUR_KERNELS))
    blur_sigma = rng.uniform(*BLUR_SIGMA)
    out = cv2.GaussianBlur(out, (ksize, ksize), blur_sigma)  # ③
    out, n_scratch = add_scratches(out, rng)             # ④
    out, noise_sigma, sp_density = add_noise(out, rng)   # ⑤
    h, w = out.shape[:2]
    return out, {
        "输出宽": w, "输出高": h,
        "模糊核": ksize, "模糊sigma": round(blur_sigma, 2),
        "划痕数": n_scratch, "噪声sigma": round(noise_sigma, 1),
        "椒盐密度": round(sp_density, 5),
    }


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="光影重生 合成退化基准集生成器（任务 3.3）")
    ap.add_argument("--input-dir", type=str, default=str(DEFAULT_BENCH),
                    help=f"现代高清图目录（默认 {DEFAULT_BENCH}）")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help=f"退化集输出目录（默认 {DEFAULT_OUT}）")
    ap.add_argument("--long-edge", type=int, default=1200,
                    help="退化图长边像素（默认 1200，模拟老照片扫描分辨率）")
    ap.add_argument("--seed", type=int, default=0,
                    help="全局种子偏移：同一文件名+偏移 => 完全一致的退化结果")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.out)
    files = sorted(p for p in in_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"[!] {in_dir} 下没有图片，请先完成任务 2.1（bench 基准图）")

    out_dir.mkdir(parents=True, exist_ok=True)
    params_csv = out_dir.parent / "degraded_params.csv"
    rows = []
    print(f"输入：{in_dir}（{len(files)} 张）\n输出：{out_dir}\n退化项：转黑白 + 长边{args.long_edge} + 模糊 + 划痕 + 噪点\n")

    for i, p in enumerate(files, 1):
        seed = (zlib.crc32(p.stem.encode()) + args.seed) % (2 ** 32)
        rng = np.random.default_rng(seed)
        img = imread_bgr(p)
        out, params = degrade(img, args.long_edge, rng)
        out_path = out_dir / f"{p.stem}.png"
        imwrite_unicode(out_path, out)
        rows.append([p.name, seed, *params.values()])
        print(f"({i:>2}/{len(files)}) {p.name:<18} -> {out_path.name:<18} "
              f"{params['输出宽']}x{params['输出高']}  划痕{params['划痕数']}条  "
              f"模糊k={params['模糊核']}  噪声σ={params['噪声sigma']}")

    with open(params_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "种子", "输出宽", "输出高", "模糊核", "模糊sigma",
                         "划痕数", "噪声sigma", "椒盐密度"])
        writer.writerows(rows)

    print(f"\n完成：{len(rows)} 张退化图 -> {out_dir}")
    print(f"参数留档：{params_csv}（测试说明文档 7.7 材料，重跑同种子可复现）")


if __name__ == "__main__":
    main()
