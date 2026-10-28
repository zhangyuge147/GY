# -*- coding: utf-8 -*-
"""
光影重生 · 评测全流水线驱动（任务 3.10）
========================================
用 3.3 合成退化集（evaluation/data/degraded/，与 bench 原图同名）跑全六步
修复流水线（去网纹→补全→超分→人脸精修→上色→考据校色直通），输出修复结果
到 evaluation/data/repaired/，供 metrics.py 计算有参考指标、3.11 盲评混排使用。

复用 restoration/pipeline.py 的加载器与 process()（同一套模型与 unload 模式），
避免重复实现；退化图与 bench 原图同名映射（degrade.py 命名约定）：
  degraded/{stem}.png  →  bench/{stem}.jpg（GT）  →  repaired/{stem}_sr_x4.png

默认每张处理完成后清理 restoration/data/steps/<stem>/ 中间结果（评测批量
不需要动画素材，省磁盘），--keep-steps 可保留。

用法（在 guangying/ 目录、guangying 环境下）：
  python evaluation/run_eval.py                       # 全部退化集（默认 4 倍）
  python evaluation/run_eval.py --scale 2             # 2 倍输出（加速调试）
  python evaluation/run_eval.py --keep-steps          # 保留 steps 中间结果
  python evaluation/run_eval.py --inpaint-thr 100     # 划痕检测更灵敏
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # evaluation/
PROJECT = ROOT.parent                           # guangying/
REST = PROJECT / "restoration"

sys.path.insert(0, str(REST))                   # 使 from pipeline import * 可用

from pipeline import load_sr, load_lama, load_facerestorer, load_colorizer
from pipeline import process, STEPS

DEGRADED = ROOT / "data" / "degraded"
REPAIRED = ROOT / "data" / "repaired"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser(
        description="光影重生·评测全流水线驱动（任务 3.10）：退化集跑六步流水线 → repaired/")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4],
                    help="输出放大倍率（默认 4）")
    ap.add_argument("--keep-steps", action="store_true",
                    help="保留 restoration/data/steps/<stem>/ 中间结果（默认清理）")
    ap.add_argument("--face-fidelity", type=float, default=0.7,
                    help="CodeFormer 保真权重（默认 0.7）")
    ap.add_argument("--inpaint-thr", type=int, default=None,
                    help="划痕检测主通道响应阈值（默认 detect_scratch 默认值 45；"
                         "2026-10-17 检测器重构后由 130 调低，折痕召回 6.5%→75-88%）")
    args = ap.parse_args()

    files = sorted(p for p in DEGRADED.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"[!] {DEGRADED} 下没有退化图，请先运行 evaluation/degrade.py")

    REPAIRED.mkdir(parents=True, exist_ok=True)
    print(f"退化集：{DEGRADED}（{len(files)} 张）")
    print(f"输出：{REPAIRED} | 倍率 x{args.scale} | 中间 steps："
          f"{'保留' if args.keep_steps else '处理完清理'}")

    upsampler = load_sr()
    lama = load_lama()
    facer = load_facerestorer(fidelity=args.face_fidelity)
    colorizer = load_colorizer()

    t_all = time.time()
    for i, p in enumerate(files, 1):
        t0 = time.time()
        print(f"({i:>2}/{len(files)}) {p.name}  ", end="", flush=True)
        final = process(p, upsampler, outscale=args.scale, lama=lama,
                        inpaint_thr=args.inpaint_thr, facer=facer,
                        colorizer=colorizer)
        out = REPAIRED / f"{p.stem}_sr_x{args.scale}.png"
        shutil.copy2(final, out)
        if not args.keep_steps:
            shutil.rmtree(STEPS / p.stem, ignore_errors=True)
        print(f"评测成品 -> {out.name}  {time.time() - t0:.1f}s")

    print(f"\n完成：{len(files)} 张 -> {REPAIRED}（总耗时 {time.time() - t_all:.1f}s）")
    print(f"下一步：python evaluation/metrics.py 计算 PSNR/SSIM/LPIPS")


if __name__ == "__main__":
    main()
