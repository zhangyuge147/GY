# -*- coding: utf-8 -*-
"""
任务 3.13 补充：人脸白色高光/泛白柔化（deglare.py，v3）
=======================================================
老照片（尤其低分辨率扫描件）常见眼睛瞳孔/鼻孔/嘴周白色高光或泛白区，属
原片反光/损伤，经超分+CodeFormer+DDColor 放大成刺眼纯白（杨靖宇 200x257
样片实测，L36）。划痕检测只认细线状结构，此类片状白斑漏接，LaMa 不处理。

v2（2026-10-18 目检返工后重写）：
  - 区域定位：不再依赖 Haar 级联精确定位眼睛/嘴巴，改用"最大人脸框 +
    面部几何比例"划出三个子 ROI：
      眼睛带   x∈[0.08,0.47] / [0.53,0.92]·w，y∈[0.28,0.52]·h
      鼻孔带   x∈[0.35,0.65]·w，            y∈[0.52,0.70]·h
      嘴巴带   x∈[0.25,0.75]·w，            y∈[0.68,0.92]·h
  - 处理策略：每个 ROI 内取 LAB L>thr 的异常亮像素 → 形态学开运算去碎屑 →
    逐连通域用 cv2.inpaint(Telea) 以周边色调重绘，膨胀半径自适应：
      眼睛带仅重绘"不贴左右边 + 填充率>0.4"的紧凑斑（保护两侧巩膜眼白）；
      鼻孔/嘴巴带重绘全部亮斑（该区本不应有刺眼白）。
  - 双点位调用：超分前（原图，radius 小、白斑未放大）+ 成品末端
    （radius 大，兜住超分/CodeFormer/DDColor 引入的高光）。
非人像图（检不出脸）零改动。

v3（2026-10-28，L43 用户目检反馈：去划痕后鼻子和嘴巴明显改变）：
  - 现象：单人深色肖像（瓜皮帽老人）修复后鼻/嘴成一片平坦白色块，
    灰胡须纹理消失。复现拆解：LaMa 脸框内改动 0px（L42 整脸剔除正常），
    全部改动来自本模块——鼻带 1077px、嘴带 519px 的"亮斑"被 Telea 整块
    平涂。这些"亮斑"实为【自然面部结构】：鼻梁镜面高光（L 达 254）、
    灰白胡须边缘——v2"该区本不应有刺眼白"的假设对高反差老人像不成立。
  - 新增三条防线（实测依据 upload_20261028_213652：毁五官的亮斑面积
    519~1400px，真实 glare 白点（虹膜反光/相纸白点）仅 ≤82px）：
    1) 分带阈值：眼带维持 L>215（L36 案例敏感度不变）；鼻/嘴带提高到
       L>235——仅"近死白"才算异常，自然高光/灰胡须（多为 215~234）不碰；
    2) 亮斑面积上限：单斑面积 > 脸框面积 0.5% 一律拒绝重绘——大亮区 =
       五官结构（鼻梁高光/胡须/颧骨），Telea 平涂必毁五官；真实白点
       远小于该上限（比例随脸框缩放，超分前后两处调用均适用）；
    3) 重绘统计透明：打印"重绘 N px（拒大亮区 M 处）"，便于目检追溯。
  - 设计原则与 L39/L42 一致：五官保真优先于"除白"——宁可漏修个别淡白斑，
    不可平涂真实面部结构。
非人像图（检不出脸）零改动。
"""
import cv2
import numpy as np

_FACE_CASC = None

# 面部几何 ROI（人脸框归一化比例）：(x0, y0, x1, y1)
_ROI_EYES_L = (0.08, 0.28, 0.47, 0.52)
_ROI_EYES_R = (0.53, 0.28, 0.92, 0.52)
_ROI_NOSE = (0.35, 0.52, 0.65, 0.70)
_ROI_MOUTH = (0.25, 0.68, 0.75, 0.92)

# v3 常量：分带阈值 + 亮斑面积上限（见模块 docstring 实测依据）
DEF_THR_EYES = 215      # 眼带：L>215 亮斑（L36 虹膜反光案例，敏感度不变）
DEF_THR_FACE = 235      # 鼻/嘴带：仅近死白（L≥235），自然高光/灰胡须不碰
DEF_MAX_BLOB_FRAC = 0.005   # 单斑面积 ≤ 脸框面积 0.5%，超过=五官结构，拒绝重绘


def _face_casc():
    global _FACE_CASC
    if _FACE_CASC is None:
        _FACE_CASC = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE_CASC


def _max_face(gray):
    fc = _face_casc()
    faces = fc.detectMultiScale(gray, 1.05, 6, minSize=(30, 30))
    if not len(faces):
        faces = fc.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
    if not len(faces):
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _roi_boxes(face):
    """人脸框 → 眼睛左/右、鼻孔、嘴巴四个 ROI 的绝对坐标"""
    fx, fy, fw, fh = face
    boxes = []
    for x0, y0, x1, y1 in (_ROI_EYES_L, _ROI_EYES_R, _ROI_NOSE, _ROI_MOUTH):
        boxes.append((int(fx + x0 * fw), int(fy + y0 * fh),
                      int(fx + x1 * fw), int(fy + y1 * fh)))
    return boxes


def _inpaint_roi(bgr, L, mask_full, x0, y0, x1, y1,
                 radius, thr, only_compact=False, max_blob=None):
    """对单 ROI 内 L>thr 亮斑做 Telea 重绘。
    返回 (重绘像素数, 拒绝的大亮区数)——v3：面积超 max_blob 的连通域跳过
    （大亮区=鼻梁高光/胡须等五官结构，Telea 平涂会毁五官，L43）。"""
    H, W = bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0
    zone = L[y0:y1, x0:x1]
    m = (zone > thr).astype(np.uint8)
    if not m.any():
        return 0, 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labm, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return 0, 0
    keep = np.zeros_like(m)
    hz, wz = m.shape[:2]
    n_rej = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 3:
            continue
        if max_blob is not None and area > max_blob:
            n_rej += 1            # L43：大亮区=五官结构，拒绝重绘
            continue
        if only_compact:                     # 眼睛：只要紧凑斑，跳过巩膜细长带
            if bw * bh > 0 and area / (bw * bh) < 0.4:
                continue
            if x <= int(wz * 0.10) or x + bw >= int(wz * 0.90):
                continue
        keep[labm == i] = 1
    if not keep.any():
        return 0, n_rej
    dil = cv2.dilate(keep, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    mask_full[y0:y1, x0:x1] = cv2.max(mask_full[y0:y1, x0:x1], dil)
    return int(keep.sum()), n_rej


def remove_face_glare(bgr: np.ndarray, radius: int = 3, thr: int = DEF_THR_EYES,
                      thr_face: int = DEF_THR_FACE,
                      max_blob_frac: float = DEF_MAX_BLOB_FRAC):
    """人脸白色高光/泛白柔化：眼/鼻/嘴几何 ROI 内亮斑 Telea 重绘。无脸直通。
    v3（L43）：眼带 L>thr(215)；鼻/嘴带 L>thr_face(235)；单斑面积 >
    max_blob_frac×脸框面积(0.5%) 一律拒绝——大亮区是五官结构不是白斑。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face = _max_face(gray)
    if face is None:
        return bgr
    out = bgr.copy()
    H, W = out.shape[:2]
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.int16)
    mask_full = np.zeros((H, W), np.uint8)
    max_blob = max_blob_frac * face[2] * face[3]    # 脸框面积 → 单斑面积上限
    n_proc = 0
    n_rej = 0
    for i, (x0, y0, x1, y1) in enumerate(_roi_boxes(face)):
        only_compact = (i < 2)               # 眼睛带保护巩膜
        band_thr = thr if i < 2 else thr_face
        px, rej = _inpaint_roi(out, L, mask_full, x0, y0, x1, y1,
                               radius, band_thr, only_compact, max_blob)
        n_proc += px
        n_rej += rej
    if mask_full.any():
        out = cv2.inpaint(out, mask_full, radius, cv2.INPAINT_TELEA)
    print(f"    [deglare] 人脸框 {face[2]}x{face[3]}，重绘亮斑像素 {n_proc} 个"
          f"（拒大亮区 {n_rej} 处，保五官）")
    return out
