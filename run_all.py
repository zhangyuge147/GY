# -*- coding: utf-8 -*-
"""
阶段5.5：全自动流水线 run_all.py
一条命令：旧照片 → 阶段3修复 → 讲解词 → TTS 语音 → 5.4 视频合成。

依赖前置：
  - 激活 guangying 环境（pipeline 需要 torch/cu128；本脚本用 sys.executable 传递给子进程，保证同环境）
  - video.py 需要 ffmpeg：系统 PATH 或 `pip install imageio-ffmpeg`（在 guangying 环境内装）

用法（在 guangying/ 目录下）：
  python run_all.py --input restoration/data/raw/old_01.jpg \
      --person 焦裕禄 \
      --script "焦裕禄，县委书记的榜样……" \
      --subtitle "焦裕禄" --out out/old_01_final.mp4

说明：
  - 讲解词通过 --script（文本）或 --script-file（txt 文件）提供；脚本生成(4.4)暂不接入自动问答，
    因为 rag.py 是问答接口、输出含 [1][2] 引用标记，直接用于旁白需额外清洗，留待后续增强。
  - --person 仅用于 pipeline --calib 考据校色（按人物/年代匹配军服校色）。
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # guangying/
RESTORE = ROOT / "restoration"
PY = sys.executable                              # 与当前一致的 python 解释器


def run_cmd(cmd, cwd=None, desc=""):
    print(f"\n===== {desc} =====")
    print("+", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise SystemExit(f"[!] {desc} 失败，退出码 {r.returncode}")


def main():
    ap = argparse.ArgumentParser(description="阶段5.5 全自动：旧照片 → 修复 → 语音 → 视频")
    ap.add_argument("--input", type=str, required=True, help="旧照片路径")
    ap.add_argument("--script", type=str, default=None, help="讲解词文本（与 --script-file 二选一）")
    ap.add_argument("--script-file", type=str, default=None, help="讲解词 txt 文件")
    ap.add_argument("--person", type=str, default=None, help="人物名（用于 --calib 考据校色）")
    ap.add_argument("--subtitle", type=str, default=None, help="主体段底部字幕（可选）")
    ap.add_argument("--credits-file", type=str, default=None, help="片尾出处卡文本文件（可选）")
    ap.add_argument("--opening", type=str, default=str(ROOT / "opening.jpg"),
                    help="开端照片路径")
    ap.add_argument("--out", type=str, default=None, help="输出 mp4 路径（默认 out/<stem>_final.mp4）")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4], help="超分倍率")
    ap.add_argument("--face-fidelity", type=float, default=0.7, help="人脸精修保真权重")
    ap.add_argument("--no-face", action="store_true", help="跳过人脸精修")
    ap.add_argument("--no-color", action="store_true", help="跳过智能上色")
    args = ap.parse_args()

    # 讲解词来源
    script = None
    if args.script:
        script = args.script.strip()
    elif args.script_file:
        script = Path(args.script_file).read_text(encoding="utf-8").strip()
    if not script:
        raise SystemExit("[!] 请用 --script 或 --script-file 提供讲解词")

    inp = Path(args.input).resolve()
    if not inp.exists():
        raise SystemExit(f"[!] 输入图片不存在：{inp}")
    stem = inp.stem
    out = Path(args.out).resolve() if args.out else (ROOT / "out" / f"{stem}_final.mp4")

    # 1) 阶段3修复（必须在 restoration/ 下运行，模块导入依赖 cwd）
    restore_cmd = [PY, "pipeline.py", "--input", str(inp),
                   "--scale", str(args.scale), "--face-fidelity", str(args.face_fidelity)]
    if args.person:
        restore_cmd += ["--calib", args.person]
    if args.no_face:
        restore_cmd += ["--no-face"]
    if args.no_color:
        restore_cmd += ["--no-color"]
    run_cmd(restore_cmd, cwd=RESTORE, desc="阶段3 修复")

    restored = RESTORE / "data" / "steps" / stem / "07_dewhite.png"
    if not restored.exists():
        raise SystemExit(f"[!] 未找到修复结果：{restored}")

    # 2) TTS（讲解词 → 语音）
    audio = ROOT / "out" / f"{stem}_tts.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([PY, str(ROOT / "dialogue" / "tts.py"),
             "--text", script, "--out", str(audio)],
            cwd=ROOT, desc="5.2 TTS 语音")

    # 3) 视频合成
    out.parent.mkdir(parents=True, exist_ok=True)
    video_cmd = [PY, str(ROOT / "dialogue" / "video.py"),
                 "--opening", str(Path(args.opening).resolve()),
                 "--image", str(restored),
                 "--audio", str(audio),
                 "--out", str(out),
                 "--captions", script]   # 讲解词 → 逐句动态字幕（随语音显示）
    if args.subtitle:
        video_cmd += ["--subtitle", args.subtitle]
    if args.credits_file:
        video_cmd += ["--credits-file", str(Path(args.credits_file).resolve())]
    run_cmd(video_cmd, cwd=ROOT, desc="5.4 视频合成")

    print(f"\n[ok] 全自动流水线完成 → {out}")


if __name__ == "__main__":
    main()