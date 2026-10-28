# -*- coding: utf-8 -*-
"""
上色后色彩修正（color_refine.py，L40 v3 + L44，2026-10-28）
=================================================
修复流水线第五步"智能上色"的后处理（纯 CPU）：

1. 灰区补色 fill_gray_regions()（默认开启，--no-colorrefine 可关）
   DDColor 对高亮度低纹理区域（雪地/白墙/天空）与深色剪影不上色（与 L37
   "纯白区域不上色"同源机制），pd_2 实测 S<0.10 区域占 16.7%。
   v1 教训（已回退）：对全部灰区做 Telea 邻域扩散 + 3x3 膨胀，导致
   a) pd_5 天空（36% 灰区）被红旗/人群边界的暖红污染成橙红渐变；
   b) 膨胀圈替换彩色边缘像素造成边界饱和度异常增强。
   v2 策略：大连通域（>= 图像 0.5%）不做邻域扩散，改用"全局环境色调"
   （非灰像素 a/b 中位数 × 0.70 强度平铺，雪地/天空获得自然的年代照片色调，
   无边界污染）；小连通域保留 Telea 局部扩散（船体剪影等获得环境色）。
   v2.1 修正（2026-10-19）：雪地近灰（强度不可辨）、看台人群异常红，调
   AMBIENT_K 0.35→0.70、GRAY_SAT_THR 0.10→0.12、降饱和参数放宽。
   v3 修正（2026-10-28，L44）：全局环境色在"整体缺色"的图上≈中性灰，导致
   同一面墙出现"亮部有色（DDColor 直上）、过曝/阴影部近灰白"的割裂
   （实测 upload_20261028_213652 过曝白窗 S=0.03 vs 亮砖 S=0.22）。
   新增"缺色图局部环境色"规则：全局环境色 chroma < COLOR_STARVED_THR（说明
   DDColor 全图只找到很少可信色彩，全局中位数趋近无信息的中性灰）时，大灰区
   改取各自"邻近彩色像素环带"的中位 a/b（邻近同材质先染色），并设
   LOCAL_CAP chroma 上限防爆色；色彩丰富图（pd_5 人群红）维持 v2 全局
   色调不变，避免 v1 天空被红旗污染的回归。

2. 色相统一 unify_chroma()（L44 新增，默认启用）
   实测 upload_20261028_222429（960x662 半身像）：同一件上衣 a/b 标准差
   高达 17~29（正常同色衣物 3~5），hue 直方图红橙带 35% 与蓝绿带 35% 并存
   ——DDColor 颜色解码器对大块平滑材质（衣料/墙面）的空间色相抖动。
   对策：ab 通道引导滤波（He 引导滤波），引导图 = 轻度高斯模糊的 L 通道
   （模糊掉印刷网点/颗粒，保留物体亮度边界）：同一材质内部（L 相近）的
   多色斑被窗口中位统一为单一色相，材质间的亮度边界上的色彩过渡保留
   （墙/衣异色但同亮度时存在轻微互相渗透，靠小半径+强度系数抑制）。
   参数随图幅自适应（半径/引导模糊），强度 UNIFY_STRENGTH=0.8。

3. 区域降饱和 desaturate_region()（--desat-mask 显式指定，默认不启用）
   DDColor 对"庆典人群"类场景的刻板先验会把看台人群刷成大面积异常红/橙
   （pd_5 实测）。hue 实测证明无法全局区分"合法暖调"与"人群异常红"
   （pd_2 暖调 24~47° 与 pd_5 人群橙红同带共存），故采用人工 mask 区域内
   定向降饱和（与考据校色 --calib-mask 同惯例）：
   - 色带限制 hue 10°~44°（OpenCV 5~22）及 320°~360°（>=160）：
     纯红旗帜/红领章（hue<10°）自动豁免；khaki 军装（64°+）不在带内
   - 仅动高饱和像素（S>0.30），肤色（S 通常<0.30）不受影响
   - 亮度 V 同步保留，仅压缩饱和度

流水线接线（pipeline.py）：DDColor 输出后、落盘 05_color 前调用，
输出仍编号 05_color（保持 00~07 前端修复过程动画契约不变）。
"""
import cv2
import numpy as np

GRAY_SAT_THR = 0.12      # 近零饱和判定阈值（pd_2 校准：雪地/天空条带覆盖）
HUGE_FRAC = 0.005        # 大连通域判定：>= 图像面积 0.5%
AMBIENT_K = 0.70         # 大灰区环境色调强度（v2.1：0.35 太弱肉眼近灰，提至 0.7）
DESAT_SAT_THR = 0.30     # 区域降饱和的饱和度门槛（v2.1：0.45 放过中度饱和红，降为 0.3）
DESAT_VAL_THR = 0.40     # 极暗像素感知弱，不动

# v3（L44）：缺色图局部环境色
COLOR_STARVED_THR = 8.0  # 全局环境色 chroma<8（0-255 Lab 尺度）视为"整体缺色"
LOCAL_RING_MIN_PX = 500  # 环带内彩色像素下限，不足则回退全局环境色
LOCAL_CAP = 16.0         # 局部环境色 chroma 上限（防爆色；实测试验墙砖 ~14）

# L44：色相统一（引导滤波）
UNIFY_STRENGTH = 0.80    # 新 ab = strength*引导滤波结果 + (1-strength)*原 ab
UNIFY_EPS = 400.0        # 引导滤波 eps（L 0-255 尺度；越大越平滑）
UNIFY_EPS2 = 0.04        # 引导图方差占比兜底（小图方差过低时用比例项）


def fill_gray_regions(bgr: np.ndarray, sat_thr: float = GRAY_SAT_THR,
                     huge_frac: float = HUGE_FRAC, ambient_k: float = AMBIENT_K):
    """灰区补色 v3：大连通域环境色调（缺色图取局部环带色）、小连通域 Telea 扩散。
    LAB 的 a/b 通道着色，亮度 L 不动。
    返回 (补色图, (大域px, 小域px, 局部色域数))。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 1] < int(sat_thr * 255)
    if not gray.any():
        return bgr, (0, 0, 0)
    total = gray.size
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # 连通域分级：大域（天空/雪地等整片灰区）→ 环境色调；小域（剪影/缝隙）→ 扩散
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        gray.astype(np.uint8), connectivity=8)
    huge_ids = [i for i in range(1, n) if stats[i, 4] >= huge_frac * total]
    huge = np.isin(labels, huge_ids) if huge_ids else np.zeros_like(gray)
    small = gray & ~huge

    n_local = 0
    if huge.any():
        med_a = float(np.median(lab[:, :, 1][~gray]))
        med_b = float(np.median(lab[:, :, 2][~gray]))
        ambient = (med_a, med_b)
        # v3（L44）：全局环境色趋近中性灰（整体缺色）→ 大灰区改取邻近彩色环带中位色
        starved = float(np.hypot(med_a - 128, med_b - 128)) < COLOR_STARVED_THR
        if starved:
            h, w = gray.shape[:2]
            ring_k = max(9, min(h, w) // 60)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * ring_k + 1, 2 * ring_k + 1))
            chrom = lab[:, :, 1:3].astype(np.int16)
            n_local = 0
            for rid in huge_ids:
                region = (labels == rid)
                ring = cv2.dilate(
                    region.astype(np.uint8), kernel).astype(bool) & ~region & ~gray
                if int(ring.sum()) < LOCAL_RING_MIN_PX:
                    lab[:, :, 1][region] = int(round(128 + (med_a - 128) * ambient_k))
                    lab[:, :, 2][region] = int(round(128 + (med_b - 128) * ambient_k))
                    continue
                la = float(np.median(chrom[:, :, 0][ring]))
                lb = float(np.median(chrom[:, :, 1][ring]))
                c = float(np.hypot(la - 128, lb - 128))
                if c > LOCAL_CAP:                       # chroma 上限
                    la = 128 + (la - 128) * LOCAL_CAP / c
                    lb = 128 + (lb - 128) * LOCAL_CAP / c
                lab[:, :, 1][region] = int(round(128 + (la - 128) * ambient_k))
                lab[:, :, 2][region] = int(round(128 + (lb - 128) * ambient_k))
                n_local += 1
        else:
            lab[:, :, 1][huge] = int(round(128 + (med_a - 128) * ambient_k))
            lab[:, :, 2][huge] = int(round(128 + (med_b - 128) * ambient_k))
    if small.any():
        m = small.astype(np.uint8) * 255
        lab[:, :, 1] = cv2.inpaint(lab[:, :, 1], m, 5, cv2.INPAINT_TELEA)
        lab[:, :, 2] = cv2.inpaint(lab[:, :, 2], m, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), (int(huge.sum()), int(small.sum()), n_local)


def _box(img, r):
    return cv2.boxFilter(img, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1),
                         anchor=(-1, -1), normalize=True, borderType=cv2.BORDER_REFLECT)


def _guided_filter(guide, p, r, eps):
    """标准 He 引导滤波（单通道，float32）。guide/p 同形；eps 相对 guide 方差。"""
    Ig = _box(guide, r)
    mean_p = _box(p, r)
    mean_Ip = _box(guide * p, r)
    mean_II = _box(guide * guide, r)
    cov_Ip = mean_Ip - Ig * mean_p
    var_I = np.maximum(mean_II - Ig * Ig, 0.0)
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * Ig
    return _box(a, r) * guide + _box(b, r)


def unify_chroma(bgr: np.ndarray, strength: float = UNIFY_STRENGTH):
    """色相统一（L44）：ab 通道引导滤波，引导图 = 高斯模糊后的 L 通道。
    同一材质内部（L 平滑）多色斑 → 窗口均值统一；材质亮度边界 → 色彩过渡保留。
    半径/模糊随图幅自适应（对 4x 超分成品友好，小图不欠拟合）。
    返回 (统一图, 改动像素数[|Δab|>2])。"""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    r = max(8, min(64, min(h, w) // 80))
    sigma = max(2.0, min(h, w) / 500.0)
    guide = cv2.GaussianBlur(L, (0, 0), sigma)

    # eps 自适应：占引导图方差的比例 + 下限，避免小图/低对比图过强保留
    gvar = float(np.var(guide))
    eps = max(UNIFY_EPS, UNIFY_EPS2 * gvar)
    qa = _guided_filter(guide, a, r, eps)
    qb = _guided_filter(guide, b, r, eps)

    a2 = strength * qa + (1.0 - strength) * a
    b2 = strength * qb + (1.0 - strength) * b
    changed = int((np.abs(a2 - a) + np.abs(b2 - b) > 2.0).sum())
    lab[:, :, 1] = np.clip(a2 + 0.5, 0, 255).astype(np.uint8)
    lab[:, :, 2] = np.clip(b2 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), changed


def desaturate_region(bgr: np.ndarray, mask: np.ndarray,
                      sat_thr: float = DESAT_SAT_THR, factor: float = 0.3):
    """区域降饱和：mask 非零区域内，暖红/橙带（hue 8~25° 或 ≥155°）高饱和
    像素（S>0.30、V>0.40）饱和度压缩至 factor 倍。
    v2.1 色带实测修正（2026-10-19）：初版 h≤25|≥155 把 mask 内纯红（h<8，旗帜/
    肖像红框）一并降饱和（顶部纯红 25282→1167px）；hue 分布实测异常红主体在
    h8~25（52.4%）、纯红 h<8 仅 4.4%，故色带下限提至 8 豁免纯红，橙红带保持。
    S 门槛 0.30、factor 0.3（v2 的 0.45/0.6 只覆盖 8.6% mask 像素，肉眼仍异常红）。
    返回 (修正图, 降饱和像素数)。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    band = ((mask > 0)
            & (((h >= 8) & (h <= 25)) | (h >= 155))
            & (s > int(sat_thr * 255)) & (v > int(DESAT_VAL_THR * 255)))
    n = int(band.sum())
    if n == 0:
        return bgr, 0
    s2 = s.copy()
    s2[band] = (s[band] * factor).astype(np.uint8)
    hsv[:, :, 1] = s2
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), n


def refine_color(bgr: np.ndarray):
    """上色后色彩修正默认入口（L40 v3 + L44）：灰区补色 → 色相统一。
    返回 (修正图, dict(huge=大灰域px, small=小灰域px, local=局部环境色域数,
    unified=色相统一改动px))。区域降饱和按需另调 desaturate_region。"""
    out, (n_huge, n_small, n_local) = fill_gray_regions(bgr)
    out, n_unified = unify_chroma(out)
    return out, {"huge": n_huge, "small": n_small, "local": n_local,
                 "unified": n_unified}
