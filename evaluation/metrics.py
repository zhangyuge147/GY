# -*- coding: utf-8 -*-
"""
光影重生 · 修复评测指标（任务 3.10）
====================================
对任务 3.10 全流水线修复结果计算有参考指标：PSNR / SSIM / LPIPS。
比较对象：修复结果（evaluation/data/repaired/）vs 彩色原图 GT（bench/）；
同时计算"退化图 vs GT"作为修复前基线，报告修复前后提升与分场景汇总。

评测口径：
  1. 尺寸对齐：退化图与修复结果统一 resize 到 GT（bench 原图）尺寸，三张图
     同像素一一对应；PSNR/SSIM 用 RGB 三通道全尺寸计算（data_range=255）。
  2. LPIPS：AlexNet 特征距离（lpips 包），输入统一 resize 到短边 256，
     归一化到 [-1,1]（感知距离对绝对分辨率不敏感，固定 256 控制显存与耗时）。
  3. 分场景汇总：按文件名前缀 building / rural / scene / portrait 分组，
     报告均值±方差——避免平均值掩盖单图波动（局限 L14）。
  4. 三结合方案分工：本脚本 = 人工退化基准图对比（有参考指标）；
     无参考指标（MUSIQ/NIQE）与人工盲评见任务 3.11。

用法（在 guangying/ 目录、guangying 环境下）：
  python evaluation/metrics.py                        # 默认路径计算并输出 CSV
  python evaluation/metrics.py --scale 2              # 匹配 2 倍修复结果
  python evaluation/metrics.py --out data/metrics_results.csv
"""
import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent            # evaluation/
PROJECT = ROOT.parent                             # guangying/
BENCH = PROJECT / "restoration" / "data" / "bench"
DEGRADED = ROOT / "data" / "degraded"
REPAIRED = ROOT / "data" / "repaired"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LPIPS_SIZE = 256      # LPIPS 统一输入短边
CATS = ("building", "rural", "scene", "portrait")


def imread_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """缩放到与 ref 同尺寸（小于则 INTER_AREA 缩小，放大则 LANCZOS 保细节）"""
    if img.shape[:2] == ref.shape[:2]:
        return img
    interp = cv2.INTER_AREA if img.shape[0] * img.shape[1] > ref.shape[0] * ref.shape[1] \
        else cv2.INTER_LANCZOS4
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=interp)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """RGB 全通道峰值信噪比（data_range=255）"""
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """RGB 全通道结构相似度（channel_axis=2, data_range=255）"""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


# LPIPS 延迟加载（首次运行会下载 AlexNet 权重到 ~/.cache/torch/hub）
_lpips_model = None


def _lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex", verbose=False)
    return _lpips_model


def lpips_dist(a: np.ndarray, b: np.ndarray) -> float:
    """AlexNet 感知距离：输入 resize 到短边 256，归一化 [-1,1]"""
    import torch
    def to_t(x: np.ndarray):
        x = cv2.resize(x, (LPIPS_SIZE, LPIPS_SIZE))
        t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)      # 1x3xHxW
    return float(_lpips()(to_t(a), to_t(b)).item())


def category(name: str) -> str:
    for c in CATS:
        if name.startswith(c):
            return c
    return "other"


def fmt(v) -> str:
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(description="光影重生·修复评测指标（任务 3.10）")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4],
                    help="匹配哪个倍率的修复结果（默认 4）")
    ap.add_argument("--bench", type=str, default=str(BENCH), help="GT 原图目录")
    ap.add_argument("--degraded", type=str, default=str(DEGRADED), help="退化图目录")
    ap.add_argument("--repaired", type=str, default=str(REPAIRED), help="修复结果目录")
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "metrics_results.csv"),
                    help="结果 CSV 输出路径")
    args = ap.parse_args()

    bench, deg_dir, rep_dir = Path(args.bench), Path(args.degraded), Path(args.repaired)
    # 修复结果命名：{stem}_sr_x{scale}.png，以原图 stem 为 key（去 _sr_x{scale} 后缀）
    suffix_tag = f"_sr_x{args.scale}"
    rep_map = {}
    for p in rep_dir.glob("*"):
        if p.suffix.lower() in IMG_EXTS and p.name.endswith(f"{suffix_tag}.png"):
            rep_map[p.stem[: -len(suffix_tag)]] = p

    rows, groups = [], {}
    for p in sorted(deg_dir.glob("*")):
        if p.suffix.lower() not in IMG_EXTS or p.stem not in rep_map:
            continue
        stem = p.stem
        gt_path = next((bench / (stem + ext) for ext in (".jpg", ".jpeg", ".png", ".bmp")
                        if (bench / (stem + ext)).exists()), None)
        if gt_path is None:
            print(f"[!] 跳过 {stem}：bench 无原图")
            continue

        gt = imread_rgb(gt_path)
        deg = resize_like(imread_rgb(p), gt)
        rep = resize_like(imread_rgb(rep_map[stem]), gt)

        base = (psnr(deg, gt), ssim(deg, gt), lpips_dist(deg, gt))
        fix = (psnr(rep, gt), ssim(rep, gt), lpips_dist(rep, gt))
        delta = tuple(f - b for f, b in zip(fix, base))          # ΔLPIPS 为负=感知更近

        row = [stem, category(stem), *[fmt(x) for x in base],
               *[fmt(x) for x in fix], *[fmt(x) for x in delta]]
        rows.append(row)
        groups.setdefault(category(stem), []).append((base, fix))
        print(f"{stem:<18} [{row[1]:<8}] 基线 {fmt(base[0])}/{fmt(base[1])}/{fmt(base[2])}"
              f" → 修复 {fmt(fix[0])}/{fmt(fix[1])}/{fmt(fix[2])}"
              f"  ΔPSNR +{fmt(delta[0])} ΔSSIM +{fmt(delta[1])} ΔLPIPS {fmt(delta[2])}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "类别", "基线PSNR", "基线SSIM", "基线LPIPS",
                    "修复PSNR", "修复SSIM", "修复LPIPS",
                    "ΔPSNR", "ΔSSIM", "ΔLPIPS"])
        w.writerows(rows)

    # 分场景汇总（均值±方差，L14：按类别分组报告避免平均值掩盖波动）
    print("\n=== 分场景汇总（均值±方差）===")
    print(f"{'类别':<10} {'n':>3} | {'PSNR↑':>12} {'SSIM↑':>12} {'LPIPS↓':>12}"
          f" | {'ΔPSNR':>10} {'ΔSSIM':>10} {'ΔLPIPS':>10}")
    for cat in CATS + ("other",):
        data = groups.get(cat, [])
        if not data:
            continue
        n = len(data)
        mean = lambda i: statistics.mean(x[i] for x, _ in data)
        stdev = lambda i: statistics.stdev(x[i] for x, _ in data) if n > 1 else 0.0
        d_mean = lambda i: statistics.mean(x[i] - y[i] for y, x in data)
        print(f"{cat:<10} {n:>3} | {fmt(mean(0))}±{fmt(stdev(0))} {fmt(mean(1))}±{fmt(stdev(1))}"
              f" {fmt(mean(2))}±{fmt(stdev(2))} | {fmt(d_mean(0))} {fmt(d_mean(1))}"
              f" {fmt(d_mean(2))}")
    all_data = [v for vs in groups.values() for v in vs]
    if all_data:
        m = lambda i: statistics.mean(x[i] for x, _ in all_data)
        d = lambda i: statistics.mean(x[i] - y[i] for y, x in all_data)
        print(f"{'总体':<10} {len(all_data):>3} | PSNR {fmt(m(0))}  SSIM {fmt(m(1))}"
              f"  LPIPS {fmt(m(2))} | ΔPSNR {fmt(d(0))}  ΔSSIM {fmt(d(1))}"
              f"  ΔLPIPS {fmt(d(2))}")

    print(f"\n结果已写入：{out}")
    print("说明：PSNR/SSIM 全尺寸 RGB 计算；LPIPS 统一 256 边 AlexNet 特征距离；"
          "\n      ΔLPIPS 为负表示感知距离更近（修复有效）。")


if __name__ == "__main__":
    main()
