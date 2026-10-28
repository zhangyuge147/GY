# -*- coding: utf-8 -*-
"""
任务 3.7：DDColor 黑白智能上色模块
================================
修复流水线六步：去网纹 → 划痕补全 → 超分辨率 → 人脸精修 → 智能上色 → 考据校色
本模块实现第五步"智能上色"：对灰度/黑白输入做端到端自动着色，输出彩色结果。
在流水线中的位置固定为 face（人脸精修）之后、calib（考据校色）之前。

为什么选 DDColor：
  DDColor（ICCV 2023，阿里达摩院）是当前黑白上色的 SOTA 方案之一，采用
  "双解码器"结构——像素解码器保留空间细节 + 颜色解码器（可学习颜色查询 +
  多尺度 Transformer）生成自然生动的色彩，对老照片的人像肤色、军服、旗帜
  等大面积单色区域着色比早期 DeOldify 更克制、偏色更少。
  本实现采用论文作者的官方 PyTorch 移植（piddnad/DDColor，2024-2026 持续
  维护），而非 MindSpore 原版，以适配本项目的 PyTorch cu128 环境。

实现说明（工程细节三处）：
  1) 依赖补丁：DDColor 的模型代码依赖仓库内置的轻量 basicsr 子包
     （basicsr/archs/ddcolor_arch_utils/*，仅 unet/convnext/transformer/
     position_encoding 五个模块，只依赖 torch）。本项目沿用 3.6 的成功模式，
     将该子包复制进 site-packages/basicsr/archs/ddcolor_arch_utils/
     （不复制 ddcolor_arch.py，避免注册表自动扫描引入 skimage 等训练依赖）。
  2) ddcolor 包：官方 PyTorch 实现（ddcolor/model.py + pipeline.py）通过
     sys.path.append(DDColor-master) 提供（append 在 site-packages 之后，
     不遮蔽补丁版 basicsr；ddcolor 包自身的 basicsr 导入也解析到 site-packages，
     与 face_restore 的 codeformer_arch 和谐共存）。
  3) 尺寸语义：本步输入是 face/sr 输出（任意尺寸），DDColor 内部缩放到
     input_size=512 推理，再把预测的 ab 通道上采样回原尺寸与原始 L 通道
     拼接输出，因此输出与输入同尺寸、无需额外后处理。
  4) 保宽高比推理（L44，2026-10-28）：官方 ColorizationPipeline.process 对
     任意尺寸一律 cv2.resize 到 512x512（不保比例）——竖图 3840x5712 被横向
     拉伸 1.49 倍、横图 3840x2648 被纵向压扁 1.45 倍，内容畸变导致色彩先验
     抖动（实测同一件衣服 hue 直方图红橙带 35% + 蓝绿带 35% 并存，见
     局限清单 L44）。v2 探索过 letterbox 灰补边方案，但 ≤0.75 宽高比的竖图
     补边占比达 25%~33%，灰带把 DDColor 的全局色彩查询带偏（杨靖宇标准像
     背景出现品红调）；经查 DDColor 位置编码为 PositionEmbeddingSine（按
     实际特征图尺寸动态生成，尺寸无关）+ ConvNeXt 全卷积编码器，模型原生
     支持非方形输入（512x384 / 384x512 等实测通过）。最终方案：长边缩到
     input_size、短边按 32 对齐取整（≤3.6% 轻度比例误差），零补边直接
     推理，ab 上采样回原尺寸拼原始 L——与官方路径逐步同构（浮点 Lab 全链路）。

用法（guangying 环境下，在 guangying/ 目录运行）：
  python restoration/color_restore.py --image data/raw/old_xx.jpg --out x.png  # 单张调试
输出：与输入同尺寸 BGR 彩色图，stdout 打印设备与耗时。
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent          # restoration/
DD_ROOT = ROOT.parent / "DDColor-master"        # guangying/DDColor-master（官方 PyTorch 源码）
sys.path.append(str(DD_ROOT))                   # 提供 ddcolor 包；不遮蔽补丁版 basicsr

from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model  # noqa: E402

MODELS = ROOT / "models"
DEFAULT_WEIGHTS = MODELS / "ddcolor_modelscope.pt"   # DDColor-L，ModelScope 官方权重

DEF_INPUT_SIZE = 512     # DDColor 内部推理尺寸（着色质量/速度的折中）
DEF_MODEL_SIZE = "large"  # large=convnext-l（默认，画质最佳）；tiny=convnext-t（更轻）


class Colorizer:
    """DDColor 黑白上色：任意尺寸 BGR 灰度图 → 同尺寸 BGR 彩色图"""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: torch.device = None,
                 input_size: int = DEF_INPUT_SIZE, model_size: str = DEF_MODEL_SIZE):
        weights = Path(weights)
        if not weights.exists():
            raise SystemExit(
                f"[!] 权重不存在：{weights}\n"
                f"    请先下载 ddcolor 权重到 restoration/models/：\n"
                f"    curl.exe -L -C - --retry 10 --retry-all-errors -o {weights} "
                f"https://modelscope.cn/api/v1/models/damo/cv_ddcolor_image-colorization/"
                f"repo?Revision=master&FilePath=pytorch_model.pt")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._weights = weights
        self.input_size = int(input_size)
        self.model_size = model_size
        self._load()

    def _load(self):
        """（重新）加载模型到设备（unload 后下次调用自动重载）"""
        self.model = build_ddcolor_model(
            DDColor,
            model_path=str(self._weights),
            input_size=self.input_size,
            model_size=self.model_size,  # 'large' 或 'tiny'
            device=self.device,
        )
        self.pipeline = ColorizationPipeline(self.model, input_size=self.input_size,
                                             device=self.device)

    def unload(self):
        """释放显存：把模型挪回 CPU 并清引用（流水线步骤之间调用，避免多模型
        常驻超 8GB 显存；详见任务 3.7 批量回归发现的 OOM 问题）"""
        for attr in ("pipeline", "model"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """
        bgr: uint8 BGR 图（流水线中为 face/sr 输出，可为灰度或已有轻微颜色）
        返回: uint8 BGR，与输入同尺寸
        L44：保宽高比推理（官方路径为不保比例的 512x512 强拉伸）
        """
        if not hasattr(self, "pipeline"):
            self._load()                       # 上次 unload 后自动重载
        return self._process_aspect(bgr)

    def _process_aspect(self, img_bgr: np.ndarray) -> np.ndarray:
        """保宽高比推理（L44）：长边缩到 input_size、短边按 32 对齐取整，
        零补边直接推理（模型位置编码为 sine、支持非方形输入，实测验证过）；
        ab 上采样回原尺寸，与原始 L 拼接。浮点 Lab 链路与官方
        ColorizationPipeline.process 逐步同构，仅预缩放一步不同。"""
        height, width = img_bgr.shape[:2]
        S = self.input_size
        scale = S / max(height, width)

        def snap(v):
            return int(min(S, max(32, round(v * scale / 32) * 32)))

        nh, nw = snap(height), snap(width)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        small = cv2.resize(img_bgr, (nw, nh), interpolation=interp)

        img = (small / 255.0).astype(np.float32)
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]     # 官方同构
        img_gray_lab = np.concatenate(
            (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        tensor_gray_rgb = (
            torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
            .float().unsqueeze(0).to(self.device))

        with torch.no_grad():                                # 推理免梯度，省显存
            output_ab = self.model(tensor_gray_rgb).cpu()   # (1, 2, nh, nw)
        ab = output_ab[0].detach().float().numpy().transpose(1, 2, 0)
        ab_full = cv2.resize(ab, (width, height), interpolation=cv2.INTER_LINEAR)

        orig = (img_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(orig, cv2.COLOR_BGR2Lab)[:, :, :1]
        output_lab = np.concatenate((orig_l, ab_full), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        return (output_bgr * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- 主流程（调试入口）
def main():
    ap = argparse.ArgumentParser(
        description="光影重生·DDColor 黑白上色（任务3.7）：全图上色（调试入口）")
    ap.add_argument("--image", type=str, required=True, help="输入图路径（BGR）")
    ap.add_argument("--out", type=str, default=None, help="输出路径（默认 <image>_color.png）")
    ap.add_argument("--input-size", type=int, default=DEF_INPUT_SIZE,
                    help=f"DDColor 内部推理尺寸（默认 {DEF_INPUT_SIZE}）")
    ap.add_argument("--model-size", type=str, default=DEF_MODEL_SIZE, choices=["large", "tiny"],
                    help=f"模型规格：large=convnext-l（画质最佳）/ tiny=convnext-t（更轻），默认 {DEF_MODEL_SIZE}")
    args = ap.parse_args()

    from preprocess import imread_bgr, imwrite_unicode

    img_p = Path(args.image)
    if not img_p.is_absolute():
        img_p = (ROOT / img_p).resolve()
    img = imread_bgr(img_p)
    t0 = time.time()
    colorizer = Colorizer(input_size=args.input_size, model_size=args.model_size)
    result = colorizer(img)
    dt = time.time() - t0

    out_p = Path(args.out) if args.out else img_p.with_name(img_p.stem + "_color.png")
    imwrite_unicode(out_p, result)
    print(f"[OK] {img_p.name}: DDColor 上色完成 | {colorizer.device} "
          f"| {dt:.1f}s -> {out_p}")


if __name__ == "__main__":
    main()
