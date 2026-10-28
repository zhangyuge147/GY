# -*- coding: utf-8 -*-
"""
de_white.py — 发白脸部肤色恢复（L37）
=====================================
问题背景：低清老合影中，部分人脸在源片上就已过曝成纯白（高光溢出）。
DDColor 上色时对纯白区域保持 A/B=128 零饱和度（不上色），导致最终成品中
这些脸呈"惨白无血色"的僵尸感（pd_11 后排多人为典型）。

本模块定位"过曝脸斑"并做温和肤色恢复：
  1. L>阈值 的亮斑连通域 + 面积/宽高比/贴边过滤，圈出"脸斑"候选；
  2. "暗邻域"约束：脸斑周围（外扩 2 倍框）必须有足够暗像素（发/帽/衣），
     以此区分过曝人脸与真实白色物体（旗帜/窗户/白墙/白袍）；
  3. 对候选脸斑在 LAB 内压缩 L（235-255 → ~178-190）并拉向暖肤色
     (A→~142, B→~152)，mask 高斯软化边缘无缝融合。

纯 cv2/numpy，CPU 零显存。默认流水线末端执行，--no-dewhite 可关。
"""
import cv2
import numpy as np


def _dark_context_frac(lab, mask_comp, x0, y0, x1, y1, W, H, expand=2.0):
    """blob bbox 外扩 expand 倍后，暗像素(L<110)占比；脸斑周围有发/帽/衣故高"""
    bw, bh = x1 - x0, y1 - y0
    ex = max(int(bw * (expand - 1) / 2), 4), max(int(bh * (expand - 1) / 2), 4)
    cx0, cy0 = max(0, x0 - ex[0]), max(0, y0 - ex[1])
    cx1, cy1 = min(W, x1 + ex[0]), min(H, y1 + ex[1])
    region = lab[cy0:cy1, cx0:cx1]
    if region.size == 0:
        return 0.0
    return float((region < 110).mean())


def de_white_faces(bgr, l_thr=218, target_l=182, target_a=142, target_b=152,
                   min_blob_w_frac=0.0035, max_blob_w_frac=0.032,
                   aspect=(0.6, 1.8), dark_ctx_frac=0.08,
                   dark_ctx_expand=2.0, strength=0.9, verbose=False):
    """对过曝脸斑做肤色恢复。返回 (out_bgr, n_kept, n_px)
    区域带约束：脸斑只存在于人群带内（y 22%-85%、x 3%-97%），排除贴边
    窗光/背景装饰；blob 宽高比/尺寸收窄到"脸"量级，避免误伤旗帜/白袍/立柱。"""
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # 1) 亮斑候选
    m = (L > l_thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)

    x_band = (int(0.03 * W), int(0.97 * W))
    y_band = (int(0.22 * H), int(0.85 * H))
    min_area = (min_blob_w_frac * W) ** 2 * 0.6
    # 面积上限考虑宽高比：宽≤max_frac*W 且 高≤宽/最小宽高比 的椭圆脸斑
    max_area = (max_blob_w_frac * W) ** 2 / aspect[0] * 1.4
    keep = np.zeros((H, W), np.uint8)
    n_kept = n_px = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        x1, y1 = x0 + bw, y0 + bh
        # 人群带约束：排除贴边窗光/顶部背景装饰/底部地面反光
        if x0 < x_band[0] or x1 > x_band[1] or y0 < y_band[0] or y1 > y_band[1]:
            continue
        ar = bw / max(bh, 1)
        if not (aspect[0] <= ar <= aspect[1]):
            continue                        # 长条/扁条（白袍/立柱/装饰）排除
        if _dark_context_frac(L, None, x0, y0, x1, y1, W, H, dark_ctx_expand) < dark_ctx_frac:
            continue                        # 周围无暗邻域 → 背景白，非人脸
        keep[labels == i] = 255
        n_kept += 1
        n_px += int(area)

    if n_kept == 0:
        if verbose:
            print(f"[dewhite] 无过曝脸斑，直通")
        return bgr, 0, 0

    # 2) 软化掩膜（外扩 + 高斯）
    ksize = max(5, round(min(W, H) * 0.004) | 1)
    soft = cv2.GaussianBlur(keep, (ksize, ksize), 0).astype(np.float32) / 255.0

    # 3) 肤色恢复：L 压缩 + A/B 拉向暖肤
    newL = np.where(L > l_thr, target_l + (L - l_thr) * 0.22, L)
    newA = A + (target_a - A) * 0.75
    newB = B + (target_b - B) * 0.75
    w = soft * strength
    L = L * (1 - w) + newL * w
    A = A * (1 - w) + newA * w
    B = B * (1 - w) + newB * w
    # 4) 回 BGR（LAB 转回必须用 uint8，否则 OpenCV 按 L∈0-100 解释导致整体偏暗）
    lab = cv2.merge([np.clip(L, 0, 255).astype(np.uint8),
                     np.clip(A, 0, 255).astype(np.uint8),
                     np.clip(B, 0, 255).astype(np.uint8)])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if verbose:
        print(f"[dewhite] 恢复 {n_kept} 个过曝脸斑（{n_px}px，L>{l_thr} 高光区）")
    return out, n_kept, n_px
