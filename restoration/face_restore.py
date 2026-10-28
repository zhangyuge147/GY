# -*- coding: utf-8 -*-
"""
任务 3.6：CodeFormer 人脸精修模块
================================
修复流水线六步：去网纹 → 划痕补全 → 超分辨率 → 人脸精修 → 智能上色 → 考据校色
本模块实现第四步"人脸精修"：对超分结果中检测到的人脸区域做 CodeFormer 高清
重绘，再以保真权重 w 与原脸融合回贴——解决 Real-ESRGAN 对人脸"过度平滑
（塑料感）"的短板（局限清单 L11）。

为什么选 CodeFormer：
  CodeFormer（NeurIPS'22）是盲人脸修复的经典模型，通过 VQGAN codebook +
  Transformer 对退化人脸做"语义先验重绘"，对老照片常见的人脸模糊、低分辨率、
  轻微损坏恢复效果好；配合 w 参数（保真度权重）可控制"重绘 vs 原样"的折中。
  项目定稿 w≈0.7（保真为主、轻度精修，避免改变人物相貌）。

实现说明（工程细节三处）：
  1) 依赖补丁：CodeFormer 的网络结构在官方仓库内置 basicsr 的
     basicsr/archs/{vqgan_arch,codeformer_arch}.py 中。本项目将其复制进
     site-packages/basicsr/archs/（补丁版 basicsr 缺 vqgan_arch，且其顶层
     import 移除了 lpips 依赖的 losses.py——若整仓库遮蔽会让基础环境不可用）。
     本模块直接 from basicsr.archs.codeformer_arch import CodeFormer。
  2) facelib：新版官方仓库用内置 facelib（namespace package，无需安装）。
     通过 sys.path.append(CodeFormer-master) 提供；basicsr 仍用补丁版
     site-packages（append 在 site-packages 之后，不会遮蔽）。
     人脸检测/解析权重（retinaface + parsenet）预置在 CWD 相对路径
     weights/facelib/（facelib 硬编码 model_dir='weights/facelib'）。
  3) 尺寸语义：流水线中本步输入是 sr 输出（已超分），故 upscale_factor=1
     原尺寸回贴，不再放大；w=0.7 融合在 CodeFormer 网络内部完成
     （net(x, w=0.7, adain=True)）。

用法（guangying 环境下，在 guangying/ 目录运行）：
  python restoration/face_restore.py --image data/steps/xx/sr.png --out x.png  # 单张调试
输出：与输入同尺寸 BGR 图，stdout 打印检测到的人脸数与耗时。
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

ROOT = Path(__file__).resolve().parent          # restoration/
CF_ROOT = ROOT.parent / "CodeFormer-master"     # guangying/CodeFormer-master（官方源码）
sys.path.append(str(CF_ROOT))                   # 提供 facelib；不遮蔽补丁版 basicsr

from basicsr.archs.codeformer_arch import CodeFormer  # noqa: E402  补丁版（见说明 1）
from basicsr.utils import img2tensor, tensor2img  # noqa: E402
from facelib.utils.face_restoration_helper import FaceRestoreHelper  # noqa: E402

MODELS = ROOT / "models"
DEFAULT_WEIGHTS = MODELS / "codeformer.pth"

DEF_FIDELITY = 0.7   # 保真权重：越大越贴近原脸（项目定稿值）
DEF_CLEAR_FACE_PX = 25.0   # L39 大脸门控阈值：源图脸高（像素），>= 此值视为"原始信息充足"
DET_MODEL = "retinaface_resnet50"


def _merge_duplicate_faces(helper, n_first):
    """双尺度检测去重（L38）：两轮 get_face_landmarks_5 已按序 append 到
    det_faces/all_landmarks_5。保留前 n_first 个（1280 尺度，脸大更准），
    对后续（640 补漏）中与已保留框高度重合者剔除。"""
    boxes = list(helper.det_faces)
    lms = list(helper.all_landmarks_5)
    if len(boxes) <= n_first:
        return
    kept_b, kept_l = boxes[:n_first], lms[:n_first]
    for i in range(n_first, len(boxes)):
        b = boxes[i]
        dup = False
        for k in kept_b:
            # b/k: [x1,y1,x2,y2,score]
            x0, y0 = max(b[0], k[0]), max(b[1], k[1])
            x1, y1 = min(b[2], k[2]), min(b[3], k[3])
            inter = max(0, x1 - x0) * max(0, y1 - y0)
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            area_k = (k[2] - k[0]) * (k[3] - k[1])
            iou = inter / (area_b + area_k - inter + 1e-6)
            if iou > 0.45:      # 同一张脸（尺度不同框有偏移，阈值放宽）
                dup = True
                break
        if not dup:
            kept_b.append(b)
            kept_l.append(lms[i])
    helper.det_faces = kept_b
    helper.all_landmarks_5 = kept_l


def _gate_clear_faces(helper, pre, thr_px):
    """L39 清晰度门控：源图脸高 >= thr_px 的大脸原始信息充足，CodeFormer 重绘
    只会抹平真实细节（皱纹/胡须/结构），用户目检 pd_4 反馈"修复后不如原黑白图
    清晰"即为此类脸被过度重绘所致——直接剔除不重绘（直通保留超分脸）。
    仅保留源图脸高 < thr_px 的小脸（信息不足）交由 CodeFormer 先验补全。
    pre 为超分前原尺度图（pipeline 传入 02_inpaint 结果）。"""
    s = pre.shape[1] / float(helper.input_img.shape[1])
    kept_b, kept_l = [], []
    for b, lm in zip(helper.det_faces, helper.all_landmarks_5):
        if (b[3] - b[1]) * s < thr_px:
            kept_b.append(b)
            kept_l.append(lm)
    helper.det_faces = kept_b
    helper.all_landmarks_5 = kept_l


def detect_source_faces(bgr: np.ndarray, device: torch.device = None):
    """L39 源图尺度人脸检测（流水线共用入口）：复用双尺度检测合并逻辑
    （1280 主尺度 + 640 补漏，眼距阈值 2px），返回检测框列表
    [[x1,y1,x2,y2], ...]（输入图坐标，float）。无脸返回 []。
    用途：多人照片的去噪豁免 / 划痕 mask 剔除 / 超分后轻锐化三处共用人脸框。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = bgr.shape[1]
    helper = FaceRestoreHelper(
        1, face_size=512, crop_ratio=(1, 1),
        det_model=DET_MODEL, save_ext='png',
        use_parse=True, device=device)
    helper.read_image(bgr)
    # read_image 对短边<512 的图会上采样——框坐标需换算回输入坐标系
    s = helper.input_img.shape[1] / float(w)
    n_first = helper.get_face_landmarks_5(
        only_center_face=False, resize=1280, eye_dist_threshold=2)
    if n_first > 0:
        helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=2)
        _merge_duplicate_faces(helper, n_first)
    return [[b[0] / s, b[1] / s, b[2] / s, b[3] / s] for b in helper.det_faces]


class FaceRestorer:
    """CodeFormer 人脸精修：sr 结果 → 人脸重绘 + w 保真融合回贴（同尺寸 BGR）"""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: torch.device = None,
                 fidelity: float = DEF_FIDELITY, clear_face_thr: float = DEF_CLEAR_FACE_PX):
        weights = Path(weights)
        if not weights.exists():
            raise SystemExit(
                f"[!] 权重不存在：{weights}\n"
                f"    请先下载 codeformer.pth 到 restoration/models/：\n"
                f"    curl.exe -L -C - --retry 10 --retry-all-errors -o {weights} "
                f"https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.fidelity = float(fidelity)
        self.clear_face_thr = float(clear_face_thr)   # L39 清晰度门控阈值（源图像素）
        self._weights = weights
        self._load()

    def _load(self):
        """（重新）加载网络到设备（unload 后下次调用自动重载）"""
        net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                         connect_list=['32', '64', '128', '256'])
        ckpt = torch.load(str(self._weights), map_location=self.device)
        net.load_state_dict(ckpt['params_ema'])
        net.eval()
        self.net = net.to(self.device)

    def unload(self):
        """释放显存：把网络挪回 CPU 并清引用（流水线步骤之间调用，避免多模型
        常驻超 8GB 显存；详见任务 3.7 批量回归发现的 OOM 问题）"""
        if hasattr(self, "net"):
            self.net = self.net.to("cpu")
            del self.net
        torch.cuda.empty_cache()

    def __call__(self, bgr: np.ndarray, pre: np.ndarray = None) -> np.ndarray:
        """
        bgr: uint8 BGR 图（流水线中为 sr 输出，已超分）
        pre: 超分前原尺度图（可选，L39 清晰度门控输入；None 时不做门控）
        返回: uint8 BGR，与输入同尺寸；未检测到人脸时原样返回（直通）
        """
        if not hasattr(self, "net"):
            self._load()                       # 上次 unload 后自动重载
        h, w = bgr.shape[:2]
        face_helper = FaceRestoreHelper(
            1,                          # upscale_factor=1：输入已超分，仅人脸重绘不放大
            face_size=512,
            crop_ratio=(1, 1),
            det_model=DET_MODEL,
            save_ext='png',
            use_parse=True,             # 语义解析 mask：只回贴人脸区域，背景不动
            device=self.device)
        face_helper.read_image(bgr)
        # 双尺度检测合并（L38，2026-10-18）：多人照片小脸在 resize=640 缩图后眼距<5px
        # 被 eye_dist_threshold 过滤直通（pd_11 实测 0/23 张），改为 1280 主检测 +
        # 640 补漏，取并集去重，最大化小脸覆盖。
        n_first = face_helper.get_face_landmarks_5(
            only_center_face=False, resize=1280, eye_dist_threshold=2)
        if n_first > 0:
            n_second = face_helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=2)
            _merge_duplicate_faces(face_helper, n_first)
        if pre is not None:
            # L39 清晰度门控：源图大脸直通（不重绘），仅保留小脸重绘
            _gate_clear_faces(face_helper, pre, self.clear_face_thr)
        num = len(face_helper.all_landmarks_5)
        if num == 0:
            return bgr                    # 无脸直通（风景/器物类素材）
        face_helper.align_warp_face()

        for cropped_face in face_helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.net(face_t, w=self.fidelity, adain=True)[0]
                restored = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
                restored = restored.astype(np.uint8)
            face_helper.add_restored_face(restored, cropped_face)

        face_helper.get_inverse_affine(None)
        return face_helper.paste_faces_to_input_image(upsample_img=None, draw_box=False)


# ---------------------------------------------------------------- 主流程（调试入口）
def main():
    ap = argparse.ArgumentParser(
        description="光影重生·CodeFormer 人脸精修（任务3.6）：人脸重绘 + w 保真融合（调试入口）")
    ap.add_argument("--image", type=str, required=True, help="输入图路径（BGR）")
    ap.add_argument("--out", type=str, default=None, help="输出路径（默认 <image>_face.png）")
    ap.add_argument("--fidelity", type=float, default=DEF_FIDELITY,
                    help=f"保真权重 0-1（越大越贴近原脸，默认 {DEF_FIDELITY}）")
    args = ap.parse_args()

    from preprocess import imread_bgr, imwrite_unicode

    img_p = Path(args.image)
    if not img_p.is_absolute():
        img_p = (ROOT / img_p).resolve()
    img = imread_bgr(img_p)
    t0 = time.time()
    restorer = FaceRestorer(fidelity=args.fidelity)
    result = restorer(img)
    dt = time.time() - t0

    out_p = Path(args.out) if args.out else img_p.with_name(img_p.stem + "_face.png")
    imwrite_unicode(out_p, result)
    print(f"[OK] {img_p.name}: 人脸精修完成（w={args.fidelity}）| {restorer.device} "
          f"| {dt:.1f}s -> {out_p}")


if __name__ == "__main__":
    main()
