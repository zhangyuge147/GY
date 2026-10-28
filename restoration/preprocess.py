# -*- coding: utf-8 -*-
"""
任务 3.4：前置处理模块——印刷网纹去除 + 去噪
==============================================
修复流水线六步之首：去网纹 → 划痕补全 → 超分辨率 → 人脸精修 → 智能上色 → 考据校色

原理
----
一、去网纹（频域陷波，检测驱动）：
  印刷品/画册扫描件的半调网纹（halftone moiré）在频谱中表现为中心直流
  分量之外的成对尖锐亮点。处理流程：
    1) 亮度通道（YCrCb 的 Y）加 Hann 窗做 FFT——窗抑制边界不连续造成的频谱泄漏
    2) 对数幅度谱减去 31x31 均值背景谱 → 残差
    3) 残差 > med + sens·σ（MAD 稳健估计）且位于有效区（排除中心主体低频
       保护圈与最外 Nyquist 边缘）→ 候选峰
    4) 局部极大筛选（5x5 邻域残差最大）→ 只留尖锐峰，排除宽带缓变泄漏
    5) 中心对称性验证：半调网纹峰成对出现（共轭对称），孤峰不可信
    6) 峰数上限 100：数量爆炸说明检测不可信（复杂纹理/噪声），跳过
    7) 通过后在【不加窗】谱上做高斯陷波（峰处压零、3σ 外恢复）→ 反 FFT
  未检出显著峰时图像原样直通——网页压缩照片/无网纹素材不做任何频域处理，
  避免误伤画质。（滤波不用加窗谱：Hann 窗趋零的角部去窗补偿会放大噪声）
  网纹主要表现为亮度调制，只处理 Y 通道；本项目素材以黑白/低饱和老照片为主。
二、去噪（空域非局部均值）：
  cv2.fastNlMeansDenoisingColored 温和去噪（默认 h=5，保边），去除胶片
  颗粒与压缩噪声，为后续划痕补全（LaMa）与超分（Real-ESRGAN）提供干净输入。

用法（guangying 环境下，在 guangying/ 目录运行）：
  python restoration/preprocess.py --all                        # 处理 data/raw 全部
  python restoration/preprocess.py --input data/raw/old_03.jpg  # 单张
  python restoration/preprocess.py --all --denoise 5 --moire-sens 5.0
输出：data/steps/<stem>/01_deraw.png（流水线中间结果，前端 6.4b 动画数据源）
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent          # restoration/
DATA = ROOT / "data"
RAW, STEPS = DATA / "raw", DATA / "steps"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 默认参数（对 15 张网页压缩老照片与合成网纹测试标定的经验值）
DEF_SENS = 5.0        # 峰显著性倍数（残差 MAD 的 σ 倍数）：越小越灵敏
DEF_DENOISE = 5       # NL-means 亮度去噪强度 h；0 = 跳过去噪
CENTER_FRAC = 0.08    # 频域中心保护圈（占 rmax 比例）：图像主体低频不陷波
EDGE_FRAC = 0.02      # 最外边缘排除：Nyquist 附近噪声不可信
NOTCH_FRAC = 0.012    # 陷波高斯 σ（占 rmax 比例）：峰周围 3σ 区域压零
MAX_PEAKS = 100       # 峰数上限：超过视为检测不可信，整图直通


# ---------------------------------------------------------------- IO（中文路径安全）
def imread_bgr(path: Path):
    """读图 + EXIF 方向转正 → BGR（与 evaluation/degrade.py 同一套读法）"""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def imwrite_unicode(path: Path, img):
    """中文路径安全写图（cv2.imwrite 不支持中文路径）"""
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"图像编码失败：{path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------- 去网纹
def remove_moire(bgr, sens=DEF_SENS):
    """
    检测并去除半调印刷网纹；返回 (处理后BGR, 峰坐标列表)。
    未检出显著峰时原图直通（列表为空），绝不盲目陷波。
    """
    h, w = bgr.shape[:2]
    if min(h, w) < 256:                # 太小的图频域分辨率不足，直通
        return bgr, []

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0].astype(np.float32) / 255.0

    # ---- 阶段一：加窗谱检测峰（Hann 窗抑制边界泄漏，让网纹峰干净可辨）----
    win = (np.hanning(h).astype(np.float32)[:, None]
           * np.hanning(w).astype(np.float32)[None, :])
    spec = np.fft.fftshift(np.fft.fft2(y * win))
    mag = np.log1p(np.abs(spec)).astype(np.float32)
    del spec
    bg = cv2.blur(mag, (31, 31))       # 谱背景估计（均值滤波，峰占 1 bin 影响可忽略）
    resid = mag - bg
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    if mad < 1e-6:
        return bgr, []
    thr = med + sens * 1.4826 * mad    # MAD→σ 换算系数 1.4826

    # 有效区：中心保护圈外、最外边缘内
    cy, cx = h / 2.0, w / 2.0
    rmax = min(cy, cx)
    gy = np.arange(h, dtype=np.float32) - cy
    gx = np.arange(w, dtype=np.float32) - cx
    r = np.hypot(gy[:, None], gx[None, :])
    valid = (r > CENTER_FRAC * rmax) & (r < (1.0 - EDGE_FRAC) * rmax)
    del r

    cand = (resid > thr) & valid
    if not cand.any():
        return bgr, []

    # 局部极大筛选：只要尖锐峰（宽带缓变的泄漏残留被排除）
    dil = cv2.dilate(resid, np.ones((5, 5), np.uint8))
    peak_mask = cand & (resid >= dil - 1e-3)
    ys, xs = np.nonzero(peak_mask)
    if len(ys) == 0:
        return bgr, []

    # 中心对称性验证：网纹峰成对出现（共轭对称），孤峰不可信
    half = med + 0.5 * (thr - med)     # 对称点残差需过半阈值
    keep = []
    for py, px in zip(ys.tolist(), xs.tolist()):
        sy, sx = int(2 * cy - py), int(2 * cx - px)
        if 0 <= sy < h and 0 <= sx < w and resid[sy, sx] >= half:
            keep.append((py, px))
    if not keep or len(keep) > MAX_PEAKS:
        # 无峰 → 无网纹；峰爆炸 → 检测不可信。两种情况都直通
        return bgr, []

    # ---- 阶段二：不加窗谱做高斯陷波 → 反 FFT ----
    sigma = NOTCH_FRAC * rmax
    R = int(3 * sigma) + 1
    # 局部陷波核：中心 0（峰处压零），3σ 外 ≈ 1（其他频率不受影响）
    ky = np.arange(2 * R + 1, dtype=np.float32) - R
    kx = np.arange(2 * R + 1, dtype=np.float32) - R
    g = 1.0 - np.exp(-(ky[:, None] ** 2 + kx[None, :] ** 2) / (2 * sigma * sigma))

    mask = np.ones((h, w), np.float32)
    for py, px in keep:
        # 峰点及其共轭对称点都陷波（对称点可能略弱于阈值，也要压制）
        for qy, qx in ((py, px), (int(2 * cy - py), int(2 * cx - px))):
            if not (0 <= qy < h and 0 <= qx < w):
                continue
            y0, y1 = max(0, qy - R), min(h, qy + R + 1)
            x0, x1 = max(0, qx - R), min(w, qx + R + 1)
            gy0, gy1 = R - (qy - y0), R + (y1 - qy)
            gx0, gx1 = R - (qx - x0), R + (x1 - qx)
            mask[y0:y1, x0:x1] *= g[gy0:gy1, gx0:gx1]

    spec_nw = np.fft.fftshift(np.fft.fft2(y))
    y2 = np.fft.ifft2(np.fft.ifftshift(spec_nw * mask)).real
    del spec_nw

    ycc2 = ycc.copy()
    ycc2[:, :, 0] = np.clip(y2 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc2, cv2.COLOR_YCrCb2BGR), keep


# ---------------------------------------------------------------- 去噪
def denoise_colored(bgr, h_lum=DEF_DENOISE, h_color=DEF_DENOISE, face_boxes=None):
    """空域非局部均值去噪（保边）；h 越大越强，老照片默认 5 温和处理。
    L39 人脸豁免（2026-10-19）：face_boxes（[x1,y1,x2,y2] 列表）提供时，人脸
    ROI 原像素直通不去噪——NL-means 会在超分前软化真实轮廓（皱纹/镜框/唇线），
    SR 无法恢复已丢信息（pd_4 实测五脸方差 -13%~-37%）。"""
    dn = cv2.fastNlMeansDenoisingColored(
        bgr, None, h_lum, h_color, templateWindowSize=7, searchWindowSize=21)
    if face_boxes:
        h, w = dn.shape[:2]
        for (x0, y0, x1, y1) in face_boxes:
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w, int(x1)), min(h, int(y1))
            if x1 <= x0 or y1 <= y0:
                continue
            dn[y0:y1, x0:x1] = bgr[y0:y1, x0:x1]   # 原始像素贴回（保留真实轮廓）
    return dn


# ---------------------------------------------------------------- 主流程
def process(path: Path, sens: float, denoise_h: int, no_moire: bool) -> None:
    img = imread_bgr(path)
    h, w = img.shape[:2]
    t0 = time.time()

    if no_moire:
        peaks, moire_msg = [], "跳过(--no-moire)"
    else:
        img, peaks = remove_moire(img, sens=sens)
        moire_msg = f"检出 {len(peaks)} 峰并陷波" if peaks else "无显著网纹，直通"

    if denoise_h > 0:
        img = denoise_colored(img, denoise_h, denoise_h)
    dn_msg = f"h={denoise_h}" if denoise_h > 0 else "关闭"

    dt = time.time() - t0
    out = STEPS / path.stem / "01_deraw.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(out, img)
    print(f"[OK] {path.name}: {w}x{h} | 网纹: {moire_msg} | 去噪: {dn_msg} | "
          f"{dt:.1f}s -> data/steps/{path.stem}/01_deraw.png")


def main():
    ap = argparse.ArgumentParser(
        description="光影重生·前置处理（任务3.4）：印刷网纹去除 + 去噪")
    ap.add_argument("--input", type=str, help="单张图片路径（相对 restoration/ 或绝对路径）")
    ap.add_argument("--all", action="store_true", help="批量处理 data/raw/ 下全部图片")
    ap.add_argument("--denoise", type=int, default=DEF_DENOISE,
                    help=f"NL-means 去噪强度 h，0=关闭（默认 {DEF_DENOISE}）")
    ap.add_argument("--moire-sens", type=float, default=DEF_SENS,
                    help=f"网纹峰检测灵敏度（MAD σ 倍数，越小越灵敏，默认 {DEF_SENS}）")
    ap.add_argument("--no-moire", action="store_true", help="跳过去网纹（只去噪）")
    args = ap.parse_args()

    if not args.all and not args.input:
        ap.error("请指定 --input <图片路径> 或 --all")

    if args.all:
        files = sorted(p for p in RAW.glob("*") if p.suffix.lower() in IMG_EXTS)
        if not files:
            raise SystemExit(f"[!] {RAW} 下没有图片")
        print(f"前置处理 {len(files)} 张 | 网纹灵敏度 {args.moire_sens} | "
              f"去噪 h={args.denoise}{'（关）' if args.denoise == 0 else ''}")
        for i, p in enumerate(files, 1):
            print(f"({i}/{len(files)}) ", end="")
            process(p, args.moire_sens, args.denoise, args.no_moire)
    else:
        p = Path(args.input)
        if not p.is_absolute():
            p = (ROOT / args.input).resolve()
        process(p, args.moire_sens, args.denoise, args.no_moire)


if __name__ == "__main__":
    main()
