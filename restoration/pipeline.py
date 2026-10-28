# -*- coding: utf-8 -*-
"""
光影重生 · 修复流水线（M1）
==========================
当前实现：去网纹+去噪（preprocess.py，任务 3.4）→ 划痕补全（inpaint.py，任务 3.5）
→ 超分辨率 Real-ESRGAN（任务清单 3.2）→ 人脸精修（face_restore.py，任务 3.6）
→ 智能上色（color_restore.py，任务 3.7）→ 考据校色（calib_restore.py，任务 3.9）

目录约定（任务清单 3.1）：
  restoration/
  ├── pipeline.py          本文件：流水线编排入口
  ├── preprocess.py        前置处理：印刷网纹去除 + 去噪（任务 3.4）
  ├── detect_scratch.py    划痕检测：形态学 → LaMa 补全用 mask（任务 3.5）
  ├── inpaint.py           LaMa 破损补全封装（任务 3.5）
  ├── face_restore.py      CodeFormer 人脸精修封装（任务 3.6）
  ├── color_restore.py     DDColor 黑白上色封装（任务 3.7）
  ├── calib_restore.py     考据校色辅助：人工对照考据色表局部校色（任务 3.9）
  ├── download_models.py   模型权重下载（含镜像 fallback）
  ├── setup_patch.py       环境兼容补丁（basicsr × 新版 torchvision）
  ├── models/              预训练权重（RealESRGAN_x4plus.pth / big-lama.pt / codeformer.pth / ddcolor_modelscope.pt）
  ├── CodeFormer-master/   CodeFormer 官方源码（提供 facelib，见 face_restore.py）
  ├── DDColor-master/      DDColor 官方 PyTorch 源码（提供 ddcolor 包，见 color_restore.py）
  └── data/
      ├── raw/             原始输入（old_01.jpg ...）
      ├── steps/<stem>/    每步中间结果（任务 3.9 统一编号，供前端"修复过程动画"6.4b）
      │   ├── 00_input.png      原始输入
      │   ├── 01_deraw.png      去网纹+去噪
      │   ├── 02_inpaint.png    划痕补全
      │   ├── 03_sr.png         超分辨率
      │   ├── 04_face.png       人脸精修
      │   ├── 05_color.png      智能上色
      │   ├── 06_calib.png      考据校色（--calib 未启用时=05_color 直通副本）
      │   ├── 06_calib_色板.png 参考色板（--calib 启用时落盘）
      │   └── 06_calib_校色记录.txt  校色留档（局部校色时落盘）
      └── out/             最终成品

用法（在 guangying/ 目录、guangying 环境下）：
  python restoration/pipeline.py --input data/raw/old_01.jpg   # 单张
  python restoration/pipeline.py --all                         # 批量处理 data/raw 全部
  python restoration/pipeline.py --all --scale 2               # 输出 2 倍
  python restoration/pipeline.py --all --no-inpaint            # 跳过划痕补全（对照实验）
  python restoration/pipeline.py --all --no-face               # 跳过人脸精修（对照实验）
  python restoration/pipeline.py --all --face-fidelity 0.5     # 人脸精修更"重绘"
  python restoration/pipeline.py --all --no-color              # 跳过智能上色（对照实验）
  python restoration/pipeline.py --all --calib 杨靖宇           # 启用考据校色（默认直通，落盘色板）
  python restoration/pipeline.py --input xx.png --calib 焦裕禄 \
        --calib-mask m.png --calib-target 中山装旧 --calib-strength 0.5  # 局部校色
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from preprocess import denoise_colored, remove_moire, imwrite_unicode
from detect_scratch import detect_scratch, overlay_mask
from deglare import remove_face_glare
from de_white import de_white_faces

ROOT = Path(__file__).resolve().parent          # restoration/
DATA = ROOT / "data"
RAW, STEPS, OUT = DATA / "raw", DATA / "steps", DATA / "out"
MODELS = ROOT / "models"

# 流水线步骤顺序（任务 3.9 全量定义；当前已实现 deraw + sr，其余占位逐步接入）
STEP_ORDER = ["deraw", "inpaint", "sr", "face", "color", "calib"]
STEP_NAMES = {
    "deraw": "去网纹", "inpaint": "划痕补全", "sr": "超分辨率",
    "face": "人脸精修", "color": "智能上色", "calib": "考据校色",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread_rgb(path: Path):
    # cv2.imread 不支持中文路径（素材库路径含中文），改用 preprocess.imread_bgr（PIL 读图 + EXIF 转正）
    from preprocess import imread_bgr
    return imread_bgr(path)


def load_sr(tile: int = 512):
    """加载 Real-ESRGAN x4plus 超分器（懒加载 import，避免环境未装时整个脚本不可用）"""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    weights = MODELS / "RealESRGAN_x4plus.pth"
    if not weights.exists():
        raise SystemExit(f"[!] 权重不存在：{weights}\n    请先运行：python restoration/download_models.py")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    use_cuda = torch.cuda.is_available()
    # RTX 5060 Laptop 8GB 显存：1080 宽输入直接 4x 易 OOM，默认 tile=512 分块；显存富余可调大提速
    return RealESRGANer(
        scale=4, model_path=str(weights), model=model,
        tile=tile, tile_pad=10, pre_pad=0,
        half=use_cuda,  # GPU 上用 fp16 提速省显存
        device=torch.device("cuda" if use_cuda else "cpu"),
    )


def load_lama():
    """加载 LaMa 补全器（懒加载；权重缺失时打印提示并返回 None，不阻塞其余步骤）"""
    try:
        from inpaint import LamaInpainter
        return LamaInpainter()
    except SystemExit as e:
        print(f"[!] LaMa 补全未启用：{e}")
        return None


def load_facerestorer(fidelity: float = 0.7):
    """加载 CodeFormer 人脸精修器（懒加载；权重缺失时打印提示并返回 None，不阻塞其余步骤）"""
    try:
        from face_restore import FaceRestorer
        return FaceRestorer(fidelity=fidelity)
    except SystemExit as e:
        print(f"[!] 人脸精修未启用：{e}")
        return None


def load_colorizer():
    """加载 DDColor 上色器（懒加载；权重缺失时打印提示并返回 None，不阻塞其余步骤）"""
    try:
        from color_restore import Colorizer
        return Colorizer()
    except SystemExit as e:
        print(f"[!] 智能上色未启用：{e}")
        return None


def load_calibrator(person: str):
    """加载考据校色器（纯 CPU 工具，无权重依赖；人物名不在考据色表时打印提示并返回 None）"""
    try:
        from calib_restore import Calibrator
        return Calibrator(person)
    except SystemExit as e:
        print(f"[!] 考据校色未启用：{e}")
        return None


def step_sr(upsampler, img, outscale=4):
    """步骤：超分辨率。后续步骤（人脸精修/上色）以本步输出为输入"""
    out, _ = upsampler.enhance(img, outscale=outscale)
    return out


def _sharpen_face_rois(out: np.ndarray, pre: np.ndarray, face_boxes) -> np.ndarray:
    """L39 大脸轻锐化（多人照片，2026-10-19）：门控直通的大脸经 Real-ESRGAN 后
    轮廓对比度不足（用户目检 pd_4：五官轮廓不如原黑白图可辨），unsharp 恢复
    已有边缘对比度——不造新细节、无 AI 幻觉。face_boxes 为源图坐标，按
    out/pre 宽度比例换算。参数保守（sigma=2、amount=0.6）防颗粒放大。"""
    sc = out.shape[1] / float(pre.shape[1])
    h, w = out.shape[:2]
    for (x0, y0, x1, y1) in face_boxes:
        pad = int((x1 - x0) * 0.15 * sc)
        sx0, sy0 = max(0, int(x0 * sc) - pad), max(0, int(y0 * sc) - pad)
        sx1, sy1 = min(w, int(x1 * sc) + pad), min(h, int(y1 * sc) + pad)
        roi = out[sy0:sy1, sx0:sx1]
        blur = cv2.GaussianBlur(roi, (0, 0), 2.0)
        out[sy0:sy1, sx0:sx1] = cv2.addWeighted(roi, 1.6, blur, -0.6, 0)
    return out


def save_step(stem: str, tag: str, img):
    """中间结果落盘 data/steps/<stem>/（前端 6.4b"修复过程动画"的数据源）"""
    d = STEPS / stem
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{tag}.png"), img)


# ---- LaMa 补全人脸保护（L35，2026-10-18 portrait_01 实测标定） ----
# 划痕穿过眼睛时，LaMa 会"幻觉式"重绘眼睛（无虹膜/翻白眼，02_inpaint 即现），
# 超分将其固化、CodeFormer 拿到烂输入也救不回，成品眼睛"离谱"。对策：检测到
# 人脸时把眼睛 ROI 从补全 mask 剔除，带划痕的眼睛交给 CodeFormer 人脸先验重建
# （对"带划痕但结构正常"的眼睛，白名单实施前 CodeFormer 实测无此问题）。
_FACE_CASC = None
_EYE_CASC = None


def _eye_protect_cascades():
    global _FACE_CASC, _EYE_CASC
    if _FACE_CASC is None:
        _FACE_CASC = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _EYE_CASC = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")
    return _FACE_CASC, _EYE_CASC


def protect_eyes_from_mask(mask: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """把眼睛 ROI 从补全 mask 剔除（0/255，白=待补全），眼睛留给人脸精修重建。
    无脸/无眼时原样返回，非人像图零改动。"""
    fc, ec = _eye_protect_cascades()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.05, 6, minSize=(80, 80))
    if not len(faces):
        faces = fc.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    if not len(faces):
        return mask
    h, w = mask.shape[:2]
    for (fx, fy, fw, fh) in faces:
        roi = gray[fy:fy + fh, fx:fx + fw]
        eyes = ec.detectMultiScale(roi, 1.1, 5, minSize=(10, 10))
        for (ex, ey, ew, eh) in eyes:
            # 过滤误检：眼睛宽须占脸宽 15%~45%，宽高比 0.4~1.6
            if not (0.15 * fw <= ew <= 0.45 * fw and 0.4 <= eh / ew <= 1.6):
                continue
            pad_x, pad_y = max(4, ew // 4), max(4, eh // 4)   # 外扩覆盖眼周
            x0, y0 = max(0, fx + ex - pad_x), max(0, fy + ey - pad_y)
            x1 = min(w, fx + ex + ew + pad_x)
            y1 = min(h, fy + ey + eh + pad_y)
            mask[y0:y1, x0:x1] = 0
    return mask


def process(path: Path, upsampler, outscale=4, pre=True, lama=None,
            inpaint_thr=None, inpaint_mask=None, dump_mask=False,
            facer=None, colorizer=None, calibrator=None,
            calib_mask=None, calib_target=None, calib_strength=None,
            deglare=True, dewhite=True, colorrefine=True,
            desat_mask=None, desat_factor=0.6, face_gate=True, tone=None):
    """处理单张：raw 输入 → 去网纹+去噪 → 划痕补全 → 超分 → 人脸精修 →
    智能上色 → 考据校色 → steps 中间结果（00-06 统一编号） + out 成品"""
    img = imread_rgb(path)
    save_step(path.stem, "00_input", img)  # 原始输入留档，供前端对比
    # L39 人脸豁免（多人照片去噪/锐化）+ L42 单人照 mask 剔除（2026-10-21）：
    # 源图尺度检出人脸框。face_boxes 仍限多人照大脸（仅去噪豁免/超分后轻锐化
    # 两处共用，保持 L33+L35 基线避免基准集回归）；mask_faces 含所有大脸（含单
    # 人照），仅用于划痕 mask 整脸剔除——单人标准像五官边缘被 LSD 误判为划痕，
    # LaMa 重绘会破坏真实五官，整脸交回直通/CodeFormer，不脑补。
    face_boxes = []
    mask_faces = []
    if pre or lama is not None:
        try:
            from face_restore import detect_source_faces, DEF_CLEAR_FACE_PX
            raw_faces = detect_source_faces(img)
        except (Exception, SystemExit) as e:
            print(f"    [face-box] 源图人脸检测失败（豁免不生效）：{e}")
            raw_faces = []
        big_faces = [b for b in raw_faces if b[3] - b[1] >= DEF_CLEAR_FACE_PX]
        if len(raw_faces) >= 2:
            face_boxes = big_faces
        mask_faces = big_faces
    if pre:
        img, peaks = remove_moire(img)     # 检测驱动：无网纹时直通不误伤
        img = denoise_colored(img, face_boxes=face_boxes)   # 大脸 ROI 免去噪（L39）
        save_step(path.stem, "01_deraw", img)
        print(f"    [deraw] 网纹峰 {len(peaks)} 个{'（已陷波）' if peaks else '，直通'}"
              + (f"｜人脸豁免 {len(face_boxes)} 框" if face_boxes else ""))
    if deglare:
        # 白斑柔化·前置（L36 v2）：超分前在原分辨率柔掉源图白斑，避免放大
        img = remove_face_glare(img, radius=1)
    if lama is not None:
        t0 = time.time()
        if inpaint_mask is not None:
            # L6 人工复核：跳过自动检测，直接读入人工编辑后的 mask 补全。
            # 人工 mask 视为最终裁定（不再叠加眼睛保护/多人大脸豁免），具
            # 备领章/徽章等史实敏感区目检语义。mask 须按 01_deraw 同尺寸编辑。
            mp = Path(inpaint_mask)
            if not mp.is_absolute():
                mp = (ROOT / mp).resolve()
            rmask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if rmask is None:
                raise SystemExit(f"[!] 无法读取人工 mask：{mp}")
            if rmask.shape[:2] != img.shape[:2]:
                raise SystemExit(f"[!] 人工 mask 尺寸 {rmask.shape[1]}x{rmask.shape[0]} "
                                 f"与当前图 {img.shape[1]}x{img.shape[0]} 不符，"
                                 f"请按 01_deraw.png 尺寸编辑 mask")
            mask = (rmask > 0).astype(np.uint8) * 255
            if mask.any():
                img = lama(img, mask)
                save_step(path.stem, "02_inpaint", img)
                print(f"    [inpaint·人工mask] 已按复核 mask 补全 "
                      f"{time.time() - t0:.1f}s")
            else:
                print("    [inpaint·人工mask] mask 为空，直通补全")
        else:
            kw = {"thr": inpaint_thr} if inpaint_thr is not None else {}
            # stem 匹配单图豁免白名单：portrait_01 强制留主通道修折痕，其余走全局
            # max_cov 回退校验（L33）
            mask, st = detect_scratch(img, stem=path.stem, **kw)   # 自动检测划痕/折痕
            if st["kept"] == 0:
                # 无显著划痕：mask 全零，直接进入通用 dump/补全出口（下段统一处理）
                pass
            else:
                mask = protect_eyes_from_mask(mask, img)   # 眼睛 ROI 剔除，留给人脸精修（L35）
                if mask_faces:
                    # L42 大脸豁免（含单人照）：皱纹/镜框/唇线/五官边缘被 LSD 误判
                    # 为划痕，LaMa 重绘会抹掉真实轮廓（pd_4 五脸方差 -8%~-20%，
                    # 单人标准像同样五官被误检），整脸从 mask 剔除，交回直通/CodeFormer
                    mh, mw = mask.shape[:2]
                    for (fx0, fy0, fx1, fy1) in mask_faces:
                        pad = int((fx1 - fx0) * 0.10)   # 外扩 10% 覆盖脸框边缘
                        mask[max(0, int(fy0) - pad):min(mh, int(fy1) + pad),
                             max(0, int(fx0) - pad):min(mw, int(fx1) + pad)] = 0
            if dump_mask:
                # L6 目检停靠点：落盘最终 mask（已含眼保护/大脸豁免，可能全零）+
                # 红色叠加图后停止，不补全/超分/上色。人工编辑 mask_scratch.png
                # 后可用 --inpaint-mask 重跑补全。upsampler 已为 None，必须在此返回。
                d = STEPS / path.stem
                d.mkdir(parents=True, exist_ok=True)
                imwrite_unicode(d / "mask_scratch.png", mask)
                imwrite_unicode(d / "mask_scratch_overlay.png",
                                overlay_mask(img, mask))
                print(f"    [dump-mask] 已落盘 mask_scratch.png + 叠加图"
                      f"（mask 覆盖 {int(mask.sum() // 255)}px），"
                      f"人工编辑后 --inpaint-mask 重跑补全")
                if hasattr(lama, "unload"):
                    lama.unload()
                return
            if st["kept"] == 0:
                print("    [inpaint] 无显著划痕，直通")
            elif mask.any():
                img = lama(img, mask)
                save_step(path.stem, "02_inpaint", img)
                print(f"    [inpaint] 检出线状结构 {st['kept']} 段，mask 覆盖 "
                      f"{st['cov'] * 100:.2f}%，已补全 {time.time() - t0:.1f}s")
            else:
                print("    [inpaint] 划痕仅落在眼睛保护区内，直通补全")
        if hasattr(lama, "unload"):
            lama.unload()   # 释放 LaMa 显存，避免与后续超分叠加（3.7 OOM 修复）
    t0 = time.time()
    out = step_sr(upsampler, img, outscale=outscale)
    dt = time.time() - t0
    save_step(path.stem, "03_sr", out)  # 本步中间结果
    if facer is not None:
        t0 = time.time()
        out = facer(out, pre=img if face_gate else None)   # pre 供 L39 大脸直通门控；--no-face-gate 强制重绘（英雄保真）
        if face_boxes:
            out = _sharpen_face_rois(out, img, face_boxes)   # L39 直通大脸轻锐化
        save_step(path.stem, "04_face", out)
        print(f"    [face] 人脸精修完成 {time.time() - t0:.1f}s"
              + (f"（含 {len(face_boxes)} 框轻锐化）" if face_boxes else ""))
        if hasattr(facer, "unload"):
            facer.unload()   # 释放 CodeFormer 显存，避免与上色叠加（3.7 OOM 修复）
    if colorizer is not None:
        t0 = time.time()
        out = colorizer(out)            # DDColor 全图上色（face 之后、考据校色之前）
        refine_msg = ""
        if colorrefine:
            # L40 v3 + L44 上色后色彩修正（2026-10-28）：灰区补色——大灰区
            # 环境色调（缺色图改取邻近彩色环带中位色）/小灰区 Telea 扩散
            # （v1 全域扩散致 pd_5 天空被红旗边界污染，已回退）；色相统一
            # 引导滤波平滑同材质多色斑（L44）。输出仍落盘 05_color，保持
            # 00~07 动画契约不变。
            from color_refine import refine_color
            out, cs = refine_color(out)
            if cs["huge"] or cs["small"]:
                refine_msg = f"｜灰区补色 大{cs['huge']}px/小{cs['small']}px"
                if cs["local"]:
                    refine_msg += f"（缺色图局部环境色 {cs['local']} 域）"
            if cs["unified"]:
                refine_msg += f"｜色相统一 {cs['unified']}px"
        if desat_mask is not None:
            # L40 区域降饱和（--desat-mask 人工指定）：修正庆典人群大面积异常
            # 红/橙；纯红旗帜（hue<10°）与 khaki 军装（64°+）色带自动豁免
            from color_refine import desaturate_region
            out, n_desat = desaturate_region(out, desat_mask, factor=desat_factor)
            refine_msg += f"｜区域降饱和 {n_desat}px"
        if tone is not None and not tone.get("bw"):
            # L45 修复偏好色调（精简版）：在 L44 色相统一之后应用——Lab 全局
            # 一致偏移/缩放不引入同物多色；考据校色（06_calib 局部校色）在
            # 后续步骤执行，史实色覆盖用户偏好。输出仍落盘 05_color，保持
            # 00~07 动画契约不变（与 L44 同一处理模式）。
            from prompt_tone import apply_tone
            out, tmsg = apply_tone(out, tone)
            refine_msg += f"｜偏好·{tmsg}"
        print(f"    [color] 智能上色完成 {time.time() - t0:.1f}s{refine_msg}")
        save_step(path.stem, "05_color", out)
        if hasattr(colorizer, "unload"):
            colorizer.unload()   # 释放 DDColor 显存，供下一张图的超分使用
    if calibrator is not None:
        t0 = time.time()
        # 考据校色（任务 3.9）：人工对照考据色表对 05_color 做局部 LAB 色偏修正。
        # 提供 --calib-mask/--calib-target 时执行一次像素级校色并留档记录；
        # 否则直通（06_calib=05_color 副本）。两种模式均落盘参考色板供人工对照。
        from calib_restore import palette_bgr
        imwrite_unicode(STEPS / path.stem / "06_calib_色板.png",
                        palette_bgr(calibrator.person))
        before = out
        if calib_mask is not None and calib_target is not None:
            s = calibrator.strength if calib_strength is None else float(calib_strength)
            out = calibrator.calibrate(out, calib_mask, calib_target, s)
            save_step(path.stem, "06_calib", out)
            calibrator.save_report(before, out, calib_mask, calib_target, s,
                                   STEPS / path.stem / "06_calib_校色记录.txt")
            print(f"    [calib] 局部校色「{calib_target}」完成 {time.time() - t0:.1f}s")
        else:
            save_step(path.stem, "06_calib", out)   # 直通副本（06_calib=05_color）
            print(f"    [calib] 直通：已落盘 {calibrator.person} 参考色板，供人工对照校色")
        # Calibrator 纯 CPU（cv2/numpy），无显存占用，无需 unload
    if deglare:
        t0 = time.time()
        # 白斑柔化·末端（L36 v2）：兜住超分/CodeFormer/DDColor 引入的高光
        out = remove_face_glare(out)
        save_step(path.stem, "06_deglare", out)
        print(f"    [deglare] 末端白斑柔化完成 {time.time() - t0:.2f}s")
    if dewhite:
        t0 = time.time()
        # 发白脸部肤色恢复（L37）：老合影后排过曝纯白脸斑 → 柔和暖肤色
        out, n_kept, n_px = de_white_faces(out)
        save_step(path.stem, "07_dewhite", out)
        print(f"    [dewhite] 过曝脸斑肤色恢复 {n_kept} 处（{n_px}px）{time.time() - t0:.2f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    # 成品命名沿用 _sr_x{outscale}（中间结果已按 00-06 统一编号，见任务清单 3.9）
    final = OUT / f"{path.stem}_sr_x{outscale}.png"
    cv2.imwrite(str(final), out)
    h, w = img.shape[:2]
    print(f"[OK] {path.name}: {w}x{h} -> {out.shape[1]}x{out.shape[0]}  {dt:.1f}s  -> {final.relative_to(ROOT)}")
    return final


def main():
    ap = argparse.ArgumentParser(
        description="光影重生修复流水线（M1：去网纹去噪 → LaMa 划痕补全 → Real-ESRGAN 超分"
                    " → CodeFormer 人脸精修 → DDColor 智能上色）")
    ap.add_argument("--input", type=str, help="单张图片路径（相对 restoration/ 或绝对路径）")
    ap.add_argument("--all", action="store_true", help="批量处理 data/raw/ 下全部图片")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4], help="输出放大倍率（默认 4）")
    ap.add_argument("--tile", type=int, default=512, help="分块尺寸，0=不分块（默认 512，OOM 时调小）")
    ap.add_argument("--no-pre", action="store_true",
                    help="跳过前置处理（去网纹+去噪），直接补全+超分（对照实验用）")
    ap.add_argument("--no-inpaint", action="store_true",
                    help="跳过划痕补全，直接超分（对照实验用）")
    ap.add_argument("--inpaint-thr", type=int, default=None,
                    help="划痕检测响应阈值（默认 detect_scratch 的保守值 130，"
                         "调低更灵敏但误检会破坏纹理）")
    ap.add_argument("--dump-mask", action="store_true",
                    help="L6 目检停靠点：只跑到划痕 mask 落盘（mask_scratch.png + "
                         "红色叠加图）即停，不补全/超分/上色；人工编辑 mask 后用 "
                         "--inpaint-mask 重跑补全")
    ap.add_argument("--inpaint-mask", type=str, default=None,
                    help="L6 人工复核：跳过自动检测，直接用指定 mask（按 01_deraw "
                         "尺寸编辑，白=待补全)补全")
    ap.add_argument("--no-face", action="store_true",
                    help="跳过人脸精修，直接输出超分结果（对照实验用）")
    ap.add_argument("--face-fidelity", type=float, default=0.7,
                    help="CodeFormer 保真权重 0-1（越大越贴近原脸，默认 0.7，"
                         "调低则重绘更强但可能改变相貌）")
    ap.add_argument("--no-face-gate", action="store_true",
                    help="关闭 L39 大脸清晰度门控：源图脸高>=25px 的大脸（英雄标准像）"
                         "仍走 CodeFormer 重绘而非直通超分结果，配合 --face-fidelity "
                         "用于保真调参（默认门控开启，保持基准集行为不变）")
    ap.add_argument("--no-color", action="store_true",
                    help="跳过智能上色，直接输出人脸精修结果（对照实验用）")
    ap.add_argument("--prompt", type=str, default=None, metavar="偏好提示词",
                    help="L45 修复偏好（精简版）：暖调/冷调/怀旧/鲜艳/淡雅/保持黑白，"
                         "自由文本按关键词匹配，未识别直通；保持黑白=跳过智能上色。"
                         "仅影响整体色调氛围，考据校色（--calib）以史料为准")
    ap.add_argument("--no-colorrefine", action="store_true",
                    help="关闭上色后色彩修正（L40：灰区补色+亮红抑制，默认开）")
    ap.add_argument("--desat-mask", type=str, default=None,
                    help="区域降饱和 mask 路径（L40：非零=待处理区域，须与 05_color 同尺寸；"
                         "修正庆典人群大面积异常红/橙，纯红旗帜 hue<10° 自动豁免）")
    ap.add_argument("--desat-factor", type=float, default=0.6,
                    help="区域降饱和系数（默认 0.6，越小压得越狠）")
    ap.add_argument("--calib", type=str, default=None, metavar="人物名",
                    help="启用考据校色（任务3.9）：指定考据色表人物，如 杨靖宇/吉鸿昌/彭雪枫/"
                         "焦裕禄/杨贵/党旗/八路军/红旗渠场景/兰考场景；默认直通并落盘参考色板")
    ap.add_argument("--calib-mask", type=str, default=None,
                    help="校色区域 mask 路径（非零=待校色，须与上色结果 05_color 同尺寸）")
    ap.add_argument("--calib-target", type=str, default=None,
                    help="校色部位（如 军上衣/红领章/中山装旧，须在考据色表内）")
    ap.add_argument("--calib-strength", type=float, default=None,
                    help="色偏修正强度 0-1（默认 0.5，1=完全替换为目标色 a/b 偏置）")
    ap.add_argument("--no-deglare", action="store_true",
                    help="关闭人脸白斑柔化后处理（L36：眼虹膜高光/嘴周泛白，默认开）")
    ap.add_argument("--no-dewhite", action="store_true",
                    help="关闭过曝脸斑肤色恢复（L37：老合影后排惨白脸 → 暖肤色，默认开）")
    args = ap.parse_args()

    if not args.all and not args.input:
        ap.error("请指定 --input <图片路径> 或 --all")

    if args.dump_mask and args.inpaint_mask:
        ap.error("--dump-mask（目检停靠）与 --inpaint-mask（人工复核补全）互斥")

    # L6 目检停靠：只出 mask 即退，跳过 SR/人脸/上色等重模型加载，加快人工迭代
    dump_only = args.dump_mask
    # L45 修复偏好解析（模型加载前）：保持黑白 → 跳过 DDColor 加载（省显存）；
    # 未命中关键词 → tone=None 直通，与不带 --prompt 的历史行为完全一致
    tone = None
    if args.prompt and args.prompt.strip():
        from prompt_tone import parse_prompt
        tone = parse_prompt(args.prompt)
        if tone is None:
            print(f"    [tone] 提示词「{args.prompt.strip()[:24]}」未命中偏好关键词，直通")
        else:
            print(f"    [tone] 修复偏好：{tone['name']}"
                  + ("（将跳过智能上色）" if tone.get("bw") else ""))
    upsampler = None if dump_only else load_sr(tile=args.tile)
    lama = None if args.no_inpaint else load_lama()
    facer = None if (args.no_face or dump_only) else load_facerestorer(fidelity=args.face_fidelity)
    colorizer = None if (args.no_color or dump_only or bool(tone and tone.get("bw"))) \
        else load_colorizer()
    calibrator = None if not args.calib else load_calibrator(args.calib)
    calib_mask = None
    if args.calib_mask:
        mpath = Path(args.calib_mask)
        if not mpath.is_absolute():
            mpath = (ROOT / mpath).resolve()
        calib_mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if calib_mask is None:
            raise SystemExit(f"[!] 无法读取 calib mask：{mpath}")
    if bool(args.calib_mask) != bool(args.calib_target):
        ap.error("--calib-mask 与 --calib-target 需同时指定")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tone is not None and tone.get("bw"):
        color_msg = "关（偏好·保持黑白）"
    else:
        color_msg = "关(--no-color)" if colorizer is None else "开（DDColor）"
    print(f"设备：{dev} | 前置处理：{'关(--no-pre)' if args.no_pre else '开'} | "
          f"划痕补全：{'关(--no-inpaint)' if lama is None else '开（LaMa）'} | "
          f"人脸精修：{'关(--no-face)' if facer is None else '开（CodeFormer w=' + str(args.face_fidelity) + '，门控:' + ('关' if args.no_face_gate else '开') + '）'} | "
          f"智能上色：{color_msg} | "
          f"考据校色：{'关' if calibrator is None else f'开（{args.calib}，直通落色板）'} | "
          f"模型：RealESRGAN_x4plus | tile={args.tile or '关'} | 输出倍率 x{args.scale}"
          + (f" | 修复偏好：{tone['name']}" if tone is not None else "")
          + (" | 【L6 目检停靠】" if dump_only else ""))

    desat_mask = None
    if args.desat_mask:
        dpath = Path(args.desat_mask)
        if not dpath.is_absolute():
            dpath = (ROOT / dpath).resolve()
        desat_mask = cv2.imread(str(dpath), cv2.IMREAD_GRAYSCALE)
        if desat_mask is None:
            raise SystemExit(f"[!] 无法读取 desat mask：{dpath}")
    calib_kw = {"calibrator": calibrator, "calib_mask": calib_mask,
                "calib_target": args.calib_target, "calib_strength": args.calib_strength,
                "deglare": not args.no_deglare, "dewhite": not args.no_dewhite,
                "colorrefine": not args.no_colorrefine,
                "desat_mask": desat_mask, "desat_factor": args.desat_factor}
    if args.all:
        files = sorted(p for p in RAW.glob("*") if p.suffix.lower() in IMG_EXTS)
        if not files:
            raise SystemExit(f"[!] {RAW} 下没有图片")
        for i, p in enumerate(files, 1):
            print(f"({i}/{len(files)}) ", end="")
            process(p, upsampler, outscale=args.scale, pre=not args.no_pre,
                    lama=lama, inpaint_thr=args.inpaint_thr,
                    inpaint_mask=args.inpaint_mask, dump_mask=args.dump_mask,
                    facer=facer, colorizer=colorizer, face_gate=not args.no_face_gate,
                    tone=tone, **calib_kw)
    else:
        p = Path(args.input)
        if not p.is_absolute():
            p = (ROOT / args.input).resolve()
        process(p, upsampler, outscale=args.scale, pre=not args.no_pre,
                lama=lama, inpaint_thr=args.inpaint_thr,
                inpaint_mask=args.inpaint_mask, dump_mask=args.dump_mask,
                facer=facer, colorizer=colorizer, face_gate=not args.no_face_gate,
                tone=tone, **calib_kw)


if __name__ == "__main__":
    main()
