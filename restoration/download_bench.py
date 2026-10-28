# -*- coding: utf-8 -*-
"""
任务 2.1：从 Pixabay 收集 20 张现代高清图，存 restoration/data/bench/

用途：这批图将被 evaluation/degrade.py 人工"做旧"（划痕+模糊+噪点+转黑白），
再送入修复流水线，与原图对比计算 PSNR/SSIM/LPIPS——即评测方法学中的
"合成退化基准"（真实老照片没有原图，无法算参考指标）。

为什么用 Pixabay 而不是 Pexels：
  Pexels 官方已暂停新 API Key 的签发（网页端仍可正常下载，只是没有脚本 API）。
  Pixabay 的 API Key 在注册后即时发放，且图片许可是"免费商用、无需署名"，
  同样零版权风险，因此改用它作为自动化下载源。

用法：
  1. 免费注册 https://pixabay.com/ ，在 https://pixabay.com/api/docs/ 页面
     登录后即可看到自己的 API Key（即时发放，无需审核）
  2. 在 guangying/.env 写入一行：PIXABAY_API_KEY=你的Key
  3. conda activate guangying，在 guangying/ 目录下运行：
     python restoration/download_bench.py

不想注册 API 的话，也可以在 pixabay.com / unsplash.com / pexels.com 网页端
手动搜索下载，文件按 portrait_01.jpg 等命名存入同一目录，并在
bench_sources.csv 里逐张登记来源（脚本会把手动记录与自动下载记录合并）。

选图标准（API 无法自动过滤，下载后请人工剔除，删掉后重跑本脚本会自动补齐）：
  - 彩色照片（黑白/复古滤镜的不要——转黑白由 degrade.py 统一生成）
  - 长边 >= 1600px（脚本已自动过滤）
  - portrait 类必须有清晰正脸（人脸精修 CodeFormer 的评测对象）
  - 避免过度后期 / HDR / 重胶片感
"""
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # guangying/
OUT_DIR = Path(__file__).resolve().parent / "data" / "bench"
MANIFEST = OUT_DIR / "bench_sources.csv"  # 来源清单：初赛合规自查材料（任务 7.9）
BAN_FILE = OUT_DIR / "banned_urls.txt"    # 持久化淘汰 URL（与槽位解耦，防覆盖丢记忆）

API_URL = "https://pixabay.com/api/"
MIN_WIDTH = 1600  # 最小宽度：保证退化+修复后仍有评测价值
MIN_HEIGHT = 1200
PER_QUERY = 20    # 每个关键词检索数量
SLEEP = 1.0       # 下载间隔（秒），礼貌性限速

# 补齐计划（2026-10-16 起调整）：building×4 / rural×9 / scene×7 已由团队自拍覆盖
# （零版权风险，登记于 bench_sources.csv），脚本仅负责从 Pixabay 补齐 portrait——
# 自拍无人像（肖像权考量），人脸精修评测所需的 8 张正脸照走网络免费图源
PLAN = {"portrait": 8}
QUERIES = {
    "portrait": [
        "chinese man portrait", "asian woman portrait",
        "elderly asian portrait", "asian student portrait",
        "woman face closeup", "man face closeup",
    ],
    "rural": [
        "rural china", "china countryside",
        "chinese farmer field", "china village house",
    ],
    "building": [
        "chinese temple", "forbidden city",
        "great wall of china", "old chinese architecture",
    ],
}


def search_photos(query: str, key: str, retries: int = 3) -> list:
    """按关键词检索 Pixabay 图片，返回照片元数据列表；SSL 被网络重置时退避重试"""
    wait = 5
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                API_URL,
                params={
                    "key": key,
                    "q": query,
                    "image_type": "photo",
                    "orientation": "horizontal",  # 横幅更接近老照片扫描件的构图
                    "min_width": MIN_WIDTH,
                    "min_height": MIN_HEIGHT,
                    "safesearch": "true",
                    "order": "popular",
                    "per_page": PER_QUERY,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("hits", [])
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt == retries:
                print(f"  [检索失败] {query}: {type(e).__name__}（已重试 {retries} 次）")
                return []
            print(f"  [网络波动] {query}: {type(e).__name__}，{wait}s 后重试（{attempt}/{retries}）")
            time.sleep(wait)
            wait *= 3
    return []


def pick_image(hit: dict):
    """从命中结果中挑最大尺寸的原图地址，返回 (url, 宽, 高)"""
    for url_key, w_key, h_key in (
        ("imageURL", "imageWidth", "imageHeight"),      # 原图（API 专有字段）
        ("fullHDURL", "fullHDWidth", "fullHDHeight"),   # 1920 长边
        ("largeImageURL", "largeImageWidth", "largeImageHeight"),  # 1280 长边
    ):
        if hit.get(url_key):
            w = hit.get(w_key) or hit.get("imageWidth") or MIN_WIDTH
            h = hit.get(h_key) or hit.get("imageHeight") or MIN_HEIGHT
            return hit[url_key], w, h
    return None, 0, 0


def next_free_name(cat: str, done: set) -> str:
    """找当前类别第一个未被占用的文件名（支持删除中间文件后补齐）"""
    idx = 1
    while f"{cat}_{idx:02d}.jpg" in done:
        idx += 1
    return f"{cat}_{idx:02d}.jpg"


_FACE_CASCADE = None


def precheck_portrait(hit: dict) -> bool:
    """下载前用 API 缩略图预检人像：恰好 1 张足够大的正脸。
    自动过滤动物照（Haar 误检的鸡/蜥蜴多为小框）、双人合影、远景小人脸。
    预检失败仅跳过该图，不影响后续候选。"""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    thumb = hit.get("webformatURL")
    if not thumb:
        return True  # 无缩略图字段时不拦截（极少见）
    try:
        resp = requests.get(thumb, timeout=20)
        arr = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return False
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(
            gray, 1.1, 6, minSize=(int(w * 0.08), int(w * 0.08)))
        if len(faces) != 1:
            return False
        x, y, fw, fh = faces[0]
        return fw * fh >= 0.05 * w * h  # 脸占画面 ≥5%：排除远景
    except Exception:
        return False  # 缩略图拉取失败视为不合格，不浪费原图下载


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        raise SystemExit(
            "未找到 PIXABAY_API_KEY：\n"
            "  1. 注册并登录 https://pixabay.com/ ，打开 https://pixabay.com/api/docs/\n"
            "     页面顶部即可看到即时发放的 API Key（无需审核）\n"
            "  2. 在 guangying/.env 写入一行：PIXABAY_API_KEY=你的Key\n"
            "  3. 重新运行本脚本\n"
            "如不想注册 API，也可在网页端手动下载后存入 data/bench/。"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in OUT_DIR.glob("*.jpg")}  # 已有文件不重复下载
    seen_ids = set()  # 本次运行内按图片 id 去重（同一图可能命中多个关键词）

    # 读取已有来源清单（保留手动下载的记录）
    manifest = {}
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if row and row[0] != "文件名":
                    manifest[row[0]] = row

    # 淘汰机制（两层）：
    #   a) banned_urls.txt 持久化淘汰 URL，与槽位解耦——槽位记录被新图覆盖也不丢记忆
    #   b) manifest 有记录但文件已删除的图 → 转 #banned# 前缀键，同时其 URL 写入 a)
    banned_urls = set()
    if BAN_FILE.exists():
        banned_urls = {u.strip() for u in BAN_FILE.read_text(encoding="utf-8").splitlines() if u.strip()}
    for fname in [f for f in list(manifest) if not f.startswith("#banned#")
                  and not (OUT_DIR / f).exists()]:
        row = manifest.pop(fname)
        manifest["#banned#" + fname] = row
        if len(row) > 2 and row[2]:
            banned_urls.add(row[2])
    BAN_FILE.write_text("\n".join(sorted(banned_urls)) + "\n", encoding="utf-8")
    if banned_urls:
        print(f"淘汰名单：{len(banned_urls)} 张（人工删除后不再补回）")
    # 已存在文件也按 URL 排除：seen_ids 只在单次运行内去重，跨运行会重复下载同一张图
    existing_urls = {
        row[2] for fname, row in manifest.items()
        if (OUT_DIR / fname).exists() and len(row) > 2
    }

    for cat, total in PLAN.items():
        got = sum(1 for n in done if n.startswith(cat + "_"))
        for query in QUERIES[cat]:
            if got >= total:
                break
            for hit in search_photos(query, key):
                if got >= total:
                    break
                if (hit["id"] in seen_ids or hit["pageURL"] in banned_urls
                        or hit["pageURL"] in existing_urls):
                    continue
                url, w, h = pick_image(hit)
                if not url or w < MIN_WIDTH:
                    continue
                if cat == "portrait" and not precheck_portrait(hit):
                    continue  # 缩略图预检未通过：动物/双人/远景脸，直接跳过
                fname = next_free_name(cat, done)
                print(f"下载 {fname} <- {hit['pageURL']} ({w}x{h})")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                (OUT_DIR / fname).write_bytes(resp.content)
                done.add(fname)
                seen_ids.add(hit["id"])
                manifest[fname] = [
                    fname, cat, hit["pageURL"], hit["user"], w, h,
                    "Pixabay Content License（免费商用、无需署名；来源站 Pixabay）",
                ]
                got += 1
                time.sleep(SLEEP)
        print(f"[{cat}] 目标 {total} 张，当前 {got} 张")

    # 写回来源清单（#banned# 行标记为"已淘汰"，保留来源供追溯，不计入有效素材）
    with open(MANIFEST, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "类别", "来源页面", "作者", "宽(px)", "高(px)", "许可"])
        for fname in sorted(manifest):
            row = list(manifest[fname])
            if fname.startswith("#banned#"):
                row[0], row[1] = fname, "已淘汰（人工删除）"
            writer.writerow(row)

    print(f"\n完成。图片目录：{OUT_DIR}")
    print(f"来源清单：{MANIFEST}（合规自查材料，任务 7.9）")
    print("提醒：请人工剔除黑白照/复古滤镜照，删除后重跑本脚本可自动补齐到目标数量。")


if __name__ == "__main__":
    main()
