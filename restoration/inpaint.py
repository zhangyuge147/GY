# -*- coding: utf-8 -*-
"""
任务 3.5：LaMa 破损补全模块
================================
修复流水线六步：去网纹 → 划痕补全 → 超分辨率 → 人脸精修 → 智能上色 → 考据校色
本模块实现第二步"划痕补全"：mask 区域破损重绘。

为什么选 LaMa：
  LaMa（Large Mask Inpainting，CVPR'22）是图像补全的经典高分辨率模型，
  对细长破损（划痕/折痕）与语义结构（五官/衣褶）都恢复自然，社区广泛
  使用其 TorchScript 版 big-lama.pt（~196MB），无需训练即可推理。

实现说明：
  封装 simple-lama-inpainting（0.1.2，--no-deps 安装）的预处理工具，
  直接加载 big-lama.pt TorchScript 权重推理。注意两点工程细节：
  1) --no-deps 安装：该包 pin 的 Pillow==9.5/numpy<2 会破坏本环境
     （basicsr/realesrgan 依赖新版），实测其代码仅用 torch/numpy/
     cv2/PIL 的稳定 API，新版依赖完全兼容；
  2) 原版 SimpleLama 类把图 pad 到 8 的倍数推理后【不裁回原尺寸】，
     输出会比输入大最多 7px——本封装修正为裁回原尺寸。

用法（guangying 环境下，在 guangying/ 目录运行）：
  python restoration/inpaint.py --image data/steps/xx/01_deraw.png \
                                --mask  data/steps/xx/mask_scratch.png   # 单张调试
输出：与输入同尺寸 BGR 图，stdout 打印耗时；--out 可指定输出路径
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from simple_lama_inpainting.utils.util import prepare_img_and_mask

ROOT = Path(__file__).resolve().parent          # restoration/
MODELS = ROOT / "models"
DEFAULT_WEIGHTS = MODELS / "big-lama.pt"


class LamaInpainter:
    """big-lama.pt TorchScript 推理封装：BGR 图 + 0/255 mask → 补全后 BGR 图"""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: torch.device = None):
        weights = Path(weights)
        if not weights.exists():
            raise SystemExit(
                f"[!] 权重不存在：{weights}\n"
                f"    请先下载 big-lama.pt 到 restoration/models/：\n"
                f"    curl -L -C - --retry 10 --retry-all-errors -o {weights} "
                f"https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._weights = Path(weights)
        self._load()

    def _load(self):
        """（重新）加载模型到设备（unload 后下次调用自动重载）"""
        self.model = torch.jit.load(str(self._weights), map_location=self.device)
        self.model.eval()

    def unload(self):
        """释放显存：把模型挪回 CPU 并清引用（流水线步骤之间调用，避免多模型
        常驻超 8GB 显存；详见任务 3.7 批量回归发现的 OOM 问题）"""
        if hasattr(self, "model"):
            self.model = self.model.to("cpu")
            del self.model
        torch.cuda.empty_cache()

    def __call__(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        bgr:  uint8 BGR 原图（或去网纹后的中间结果）
        mask: uint8 单通道，非零区域 = 待重绘（detect_scratch 的输出）
        返回: uint8 BGR，与输入同尺寸（修正原版不裁 pad 的坑）
        """
        if not hasattr(self, "model"):
            self._load()                       # 上次 unload 后自动重载
        if bgr.shape[:2] != mask.shape[:2]:
            raise ValueError(f"图与 mask 尺寸不一致：{bgr.shape[:2]} vs {mask.shape[:2]}")
        h, w = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_t, mask_t = prepare_img_and_mask(rgb, mask, self.device)  # pad 到 8 倍数
        with torch.inference_mode():
            out = self.model(img_t, mask_t)
        out = out[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
        out = (out[:h, :w] * 255).round().astype(np.uint8)            # 裁回原尺寸
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- 主流程（调试入口）
def main():
    ap = argparse.ArgumentParser(
        description="光影重生·LaMa 补全（任务3.5）：mask 区域破损重绘（调试入口）")
    ap.add_argument("--image", type=str, required=True, help="输入图路径（BGR）")
    ap.add_argument("--mask", type=str, required=True, help="mask 路径（非零=待补全）")
    ap.add_argument("--out", type=str, default=None, help="输出路径（默认 <image>_inpainted.png）")
    args = ap.parse_args()

    from preprocess import imread_bgr, imwrite_unicode

    img_p, mask_p = Path(args.image), Path(args.mask)
    img = imread_bgr(img_p)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"[!] 无法读取 mask：{mask_p}")

    t0 = time.time()
    lama = LamaInpainter()
    result = lama(img, mask)
    dt = time.time() - t0

    out_p = Path(args.out) if args.out else img_p.with_name(img_p.stem + "_inpainted.png")
    imwrite_unicode(out_p, result)
    print(f"[OK] {img_p.name}: mask 覆盖 {(mask > 0).mean() * 100:.2f}% | "
          f"{lama.device} | {dt:.1f}s -> {out_p}")


if __name__ == "__main__":
    main()
