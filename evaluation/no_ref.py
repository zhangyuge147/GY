# -*- coding: utf-8 -*-
"""
光影重生 · 无参考质量指标（任务 3.11）
========================================
对任务 3.10 修复结果计算无参考指标 BRISQUE（自然场景统计 + SVR，越低越自然），
无需 GT 原图——补全三结合评测方案的第二环（无参考指标）。

BRISQUE 说明：任务清单要求"MUSIQ 或 NIQE"；piq 0.8.0 已移除 niqe 接口，
故以同族经典无参考指标 BRISQUE（piq.brisque，NSS 特征 + 预训练 SVR）替代，
两者同为"无参考自然度评分"族（越小越接近自然图像），评测意图一致。
权重 brisque_svm_weights.pt（115KB）已下载至 ~/.cache/torch/hub/checkpoints/。

口径：退化图（修复前）、修复结果（修复后）、GT 彩色原图分别统一 resize 到
短边 512 后计算 BRISQUE（无参考指标对绝对分辨率不敏感，固定短边控制耗时）；
报告修复前后变化与按类别汇总。GT 值仅作"自然度上界"参照（GT 为现代拍摄
彩色照，BRISQUE 最低，天然自然）。

用法（在 guangying/ 目录、guangying 环境下）：
  python evaluation/no_ref.py                     # 默认路径计算
  python evaluation/no_ref.py --out data/no_ref_results.csv
"""
import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent            # evaluation/
PROJECT = ROOT.parent                             # guangying/
BENCH = PROJECT / "restoration" / "data" / "bench"
DEGRADED = ROOT / "data" / "degraded"
REPAIRED = ROOT / "data" / "repaired"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SIDE = 512        # 统一短边（无参考指标固定分辨率控制耗时）
CATS = ("building", "rural", "scene", "portrait")


def imread_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_side(img: np.ndarray, side: int = SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= side:
        return img
    k = side / max(h, w)
    return cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def brisque(rgb: np.ndarray) -> float:
    """BRISQUE 无参考自然度（piq，越低越自然）。输入 resize 到短边 512、[0,1]"""
    import piq
    x = resize_side(rgb)
    t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return float(piq.brisque(t).item())


def category(name: str) -> str:
    for c in CATS:
        if name.startswith(c):
            return c
    return "other"


def fmt(v) -> str:
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description="光影重生·无参考指标 BRISQUE（任务 3.11）")
    ap.add_argument("--bench", type=str, default=str(BENCH), help="GT 原图目录")
    ap.add_argument("--degraded", type=str, default=str(DEGRADED), help="修复前退化图目录")
    ap.add_argument("--repaired", type=str, default=str(REPAIRED), help="修复后结果目录")
    ap.add_argument("--scale", type=int, default=4, help="修复结果倍率（命名匹配）")
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "no_ref_results.csv"),
                    help="结果 CSV 输出路径")
    args = ap.parse_args()

    bench, deg_dir, rep_dir = Path(args.bench), Path(args.degraded), Path(args.repaired)
    tag = f"_sr_x{args.scale}"
    rep_map = {p.stem[:-len(tag)]: p for p in rep_dir.glob("*")
               if p.suffix.lower() in IMG_EXTS and p.name.endswith(f"{tag}.png")}

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
        gt = brisque(imread_rgb(gt_path))
        base = brisque(imread_rgb(p))
        fix = brisque(imread_rgb(rep_map[stem]))
        row = [stem, category(stem), fmt(base), fmt(fix), fmt(gt),
               fmt(fix - base)]
        rows.append(row)
        groups.setdefault(category(stem), []).append((base, fix))
        print(f"{stem:<18} [{row[1]:<8}] 修复前 {fmt(base)} → 修复后 {fmt(fix)}"
              f"  GT {fmt(gt)}  ΔBRISQUE {fmt(fix - base)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "类别", "修复前BRISQUE", "修复后BRISQUE",
                    "GT_BRISQUE", "ΔBRISQUE"])
        w.writerows(rows)

    print("\n=== 分场景汇总（均值±方差，BRISQUE 越低越自然）===")
    for cat in CATS + ("other",):
        data = groups.get(cat, [])
        if not data:
            continue
        n = len(data)
        m = lambda i: statistics.mean(pair[i] for pair in data)
        s = lambda i: statistics.stdev(pair[i] for pair in data) if n > 1 else 0.0
        print(f"{cat:<10} {n:>3} | 前 {fmt(m(0))}±{fmt(s(0))}  后 {fmt(m(1))}±{fmt(s(1))}"
              f"  Δ {fmt(m(1) - m(0))}")
    all_data = [v for vs in groups.values() for v in vs]
    if all_data:
        m = lambda i: statistics.mean(pair[i] for pair in all_data)
        print(f"{'总体':<10} {len(all_data):>3} | 前 {fmt(m(0))}  后 {fmt(m(1))}"
              f"  Δ {fmt(m(1) - m(0))}")
    print(f"\n结果已写入：{out}\n说明：BRISQUE 越低越接近自然图像；GT 为现代彩色"
          f"原图，作为自然度上界参照。")

if __name__ == "__main__":
    main()
