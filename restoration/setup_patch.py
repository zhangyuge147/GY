# -*- coding: utf-8 -*-
"""
环境兼容性补丁：basicsr 1.4.2 × 新版 torchvision（>= 0.17）
============================================================
问题：basicsr/data/degradations.py 中使用了
      `from torchvision.transforms.functional_tensor import rgb_to_grayscale`
      该模块在新版 torchvision（本机 0.24.1）中已被移除，导致
      `from realesrgan import RealESRGANer` 时 ImportError。
修复：将导入改为 `from torchvision.transforms.functional import rgb_to_grayscale`
     （两者功能一致，官方迁移路径）。
用法：python restoration/setup_patch.py   （幂等，可重复运行；升级/重装 basicsr 后需重跑）
"""
import importlib.util
from pathlib import Path

BAD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
GOOD = "from torchvision.transforms.functional import rgb_to_grayscale"


def patch_basicsr() -> bool:
    spec = importlib.util.find_spec("basicsr")
    if spec is None:
        print("[!] 未安装 basicsr，请先：pip install realesrgan")
        return False
    target = Path(spec.origin).parent / "data" / "degradations.py"
    if not target.exists():
        print(f"[!] 未找到 {target}")
        return False
    text = target.read_text(encoding="utf-8")
    if BAD in text:
        target.write_text(text.replace(BAD, GOOD), encoding="utf-8")
        print(f"[OK] 已修补：{target}")
    elif GOOD in text:
        print("[OK] basicsr 已是修补后状态，无需处理")
    else:
        print("[?] 未找到目标导入语句（basicsr 版本可能已变化），请人工检查 " + str(target))
    return True


if __name__ == "__main__":
    patch_basicsr()
