# -*- coding: utf-8 -*-
"""
下载修复流水线所需预训练权重到 restoration/models/

用法（guangying 环境）：
  python restoration/download_models.py            # 下载全部
  python restoration/download_models.py x4plus     # 只下载文件名含 x4plus 的

已内置 GitHub 直连 + ghproxy 镜像两级来源，依次尝试；下载用 .part 临时文件，
成功后改名，避免中断产生损坏的权重文件。已存在的文件自动跳过（幂等）。
"""
import sys
import time
from pathlib import Path

import requests

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# (文件名, [下载地址列表：依次尝试])
FILES = [
    (
        "RealESRGAN_x4plus.pth",  # 通用 4x 超分，~64MB
        [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "https://mirror.ghproxy.com/https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        ],
    ),
    # 后续按需追加：
    # ("CodeFormerr.pth", [...])   人脸精修（任务 3.5）
    # ("DDColor-models.pth", [...]) 上色（任务 3.6）
    # ("big-lama.pt", [...])       划痕补全（任务 3.4）
]


def download(url: str, dst: Path, timeout: int = 60, retries: int = 4):
    """支持断点续传的下载：网络中断后重跑，从 .part 文件已有字节继续（Range 请求）"""
    part = dst.with_suffix(dst.suffix + ".part")


    for attempt in range(1, retries + 1):
        try:
            downloaded = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True, headers=headers) as r:
                if downloaded and r.status_code != 206:  # 服务器不支持续传，从头来
                    downloaded = 0
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + downloaded
                if downloaded:
                    print(f"\r  续传自 {downloaded / 1048576:.1f} MB", end="")
                mode = "ab" if downloaded else "wb"
                t0 = time.time()
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\r  下载中 {downloaded / 1048576:.1f}/{total / 1048576:.1f} MB", end="")
                print(f"  （{time.time() - t0:.0f}s）")
            part.rename(dst)  # 下载完整后才改名
            return
        except Exception as e:
            print(f"  第 {attempt}/{retries} 次尝试失败：{type(e).__name__}（已缓存 {part.stat().st_size / 1048576:.1f} MB）")
            if attempt == retries:
                raise


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for fname, urls in FILES:
        if only and only not in fname:
            continue
        dst = MODELS_DIR / fname
        if dst.exists():
            print(f"[跳过] {fname} 已存在")
            continue
        print(f"[下载] {fname}")
        for i, url in enumerate(urls, 1):
            try:
                print(f"  来源 {i}/{len(urls)}：{url}")
                download(url, dst)
                break
            except Exception as e:
                print(f"  失败：{type(e).__name__}: {e}")
        else:
            raise SystemExit(f"[!] {fname} 全部来源均失败，请手动下载后放入：{MODELS_DIR}")
    print("完成。")


if __name__ == "__main__":
    main()
