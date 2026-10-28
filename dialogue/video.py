# -*- coding: utf-8 -*-
"""
阶段5.4：视频合成 video.py
把「开端照片片头 → 修复照片 → 语音讲解」合成为一条 mp4，供阶段5全自动流水线调用。

依赖：
  - ffmpeg：优先用系统 PATH 中的 ffmpeg；没有则用 `pip install imageio-ffmpeg` 自带的二进制
  - Pillow：把任意比例图片统一到 16:9 画布（模糊背景放大 + 原图等比居中）
  - wave（标准库）：读取 wav 音频时长，避免额外依赖 ffprobe

用法（在 guangying/ 目录下运行）：
  python dialogue/video.py \
      --opening opening.jpg \
      --image restoration/data/steps/old_01/07_dewhite.png \
      --audio out/story.wav \
      --out out/video.mp4 \
      --subtitle "焦裕禄同志是县委书记的榜样。"

说明：
  - 本版主体只支持单张修复图（满足“一张旧照片 → 视频”里程碑）；六步动画多图轮播需后续扩展。
  - 字幕用 drawtext 烧录，需系统中文字体（楷体/微软雅黑）；若字幕报错可先去掉 --subtitle 跑通主体。
"""

import argparse
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # dialogue/
DEFAULT_OPENING = ROOT.parent / "opening.jpg"   # guangying/opening.jpg

# 中文字体候选（用于字幕烧录，取第一个存在的）
CJK_FONTS = [
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

DEFAULT_SIZE = (1920, 1080)


def find_ffmpeg():
    """定位 ffmpeg：系统 PATH 优先，其次 imageio-ffmpeg 自带二进制。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # 提供独立 ffmpeg 二进制，无需系统安装
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and Path(path).exists():
            return str(path)
    except ImportError:
        pass
    raise SystemExit("[!] 未找到 ffmpeg。请执行：pip install imageio-ffmpeg")


def wav_duration(wav_path):
    """读取 wav 时长（秒），用标准库 wave，无需 ffprobe。"""
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def fit_cover(src, out, size=DEFAULT_SIZE, bg_blur=30):
    """把任意比例图片放入 16:9 画布：模糊放大图铺底 + 原图等比居中。"""
    from PIL import Image, ImageFilter, ImageOps

    W, H = size
    img = Image.open(src).convert("RGB")
    bg = ImageOps.fit(img, (W, H), Image.LANCZOS)  # cover 裁剪铺满
    bg = bg.filter(ImageFilter.GaussianBlur(bg_blur))
    fg = ImageOps.contain(img, (W, H), Image.LANCZOS)  # 等比缩小到画布内
    bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    bg.save(out)
    return out


def _run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 失败：\n" + (r.stderr or r.stdout)[-2500:])
    return r


def find_cjk_font():
    """返回第一个存在的中文字体绝对路径，供复制到临时目录烧录字幕。"""
    for f in CJK_FONTS:
        if Path(f).exists():
            return f
    return None


def _escape_drawtext(text):
    """drawtext 文本转义（冒号、反斜杠、百分号等）。"""
    for a, b in (("\\", "\\\\"), (":", "\\:"), ("'", "\\'"),
                 ("%", "\\%"), (",", "\\,"), ("[", "\\["), ("]", "\\]")):
        text = text.replace(a, b)
    return text


def _drawtext_vf(subtitle, fontfile="font.ttf"):
    """构造 drawtext 滤镜片段（底部居中白字 + 黑色描边）。

    fontfile 用相对文件名（字体已由调用方复制到工作目录），
    避免 Windows 盘符冒号在 ffmpeg 滤镜里无法可靠转义的问题。
    """
    if not subtitle:
        return ""
    text = _escape_drawtext(subtitle)
    return (",drawtext=fontfile=%s:text=%s:fontcolor=white:"
            "fontsize=44:borderw=3:bordercolor=black@0.7:"
            "x=(w-text_w)/2:y=h-150" % (fontfile, text))


def _split_sentences(text, max_len=18):
    """把讲解词按句末标点切句，超长句再按 max_len 二次硬切，供逐句字幕使用。"""
    import re
    parts = re.split(r"([。！？!?；;])", text)
    raw, buf = [], ""
    for p in parts:
        if not p:
            continue
        buf += p
        if p in "。！？!?；;":
            raw.append(buf)
            buf = ""
    if buf.strip():
        raw.append(buf)
    result = []
    for s in raw:
        s = s.strip()
        while len(s) > max_len:
            result.append(s[:max_len])
            s = s[max_len:]
        if s:
            result.append(s)
    return result


def _captions_vf(sentences, total_dur, fontfile="font.ttf"):
    """逐句动态字幕：每句一个 drawtext，用 enable 按时间窗显示。

    每句时长按“字数占全文比例 × 总音频时长”近似分配；逗号是滤镜链分隔符，
    enable 表达式内的逗号必须转义（用反斜杠），平台单引号定界不可靠。
    """
    if isinstance(sentences, str):
        sentences = _split_sentences(sentences)
    if not sentences:
        return ""
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return ""
    vf, t = "", 0.0
    for s in sentences:
        dur = total_dur * len(s) / total_chars
        text = _escape_drawtext(s)
        vf += (",drawtext=fontfile=%s:text=%s:fontcolor=white:"
               "fontsize=44:borderw=3:bordercolor=black@0.7:"
               "x=(w-text_w)/2:y=h-150:enable=between(t\\,%.2f\\,%.2f)"
               % (fontfile, text, t, t + dur))
        t += dur
    return vf


def _segment(ffmpeg, image, duration, audio, out, subtitle=None,
             captions=None, fontfile=None, cwd=None):
    """把「一张图 + 时长」渲染为带（静音或真实）音轨的片段 mp4。"""
    cmd = [ffmpeg, "-y",
           "-loop", "1", "-i", str(image),
           "-t", f"{duration:.3f}"]
    if audio:
        cmd += ["-i", str(audio)]
    else:
        # 片头段合成静音音轨，保证与主体段 concat 时音轨参数一致
        cmd += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100"]

    cap_vf = (_captions_vf(captions, duration, fontfile)
              if captions else _drawtext_vf(subtitle, fontfile))
    vf = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
          "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
          "fps=30" + cap_vf)
    cmd += ["-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-shortest", str(out)]
    _run(cmd, cwd=cwd)


def _concat(ffmpeg, files, out):
    """用 concat demuxer 拼接各段（编码参数已统一，直接 -c copy）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for p in files:
            f.write(f"file '{Path(p).as_posix()}'\n")
        list_path = f.name
    try:
        _run([ffmpeg, "-y", "-f", "concat", "-safe", "0",
              "-i", list_path, "-c", "copy", str(out)])
    finally:
        Path(list_path).unlink(missing_ok=True)


def _make_credits_image(lines, font_path, size=DEFAULT_SIZE, out_path=None):
    """用 PIL 生成片尾出处卡：深色底，首行金色标题、其余白字居中。"""
    from PIL import Image, ImageDraw, ImageFont
    W, H = size
    img = Image.new("RGB", size, (22, 17, 17))
    d = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(font_path, 60)
    body_font = ImageFont.truetype(font_path, 38)
    line_h = 68
    total_h = sum(line_h for ln in lines)
    y = max(120, (H - total_h) // 2)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            y += 30
            continue
        if i == 0:
            f, color = title_font, (255, 200, 0)
        else:
            f, color = body_font, (238, 238, 238)
        tw = d.textlength(line, font=f)
        d.text(((W - tw) / 2, y), line, font=f, fill=color)
        y += line_h
    img.save(out_path)
    return out_path


def render(opening, image, audio, out, opening_duration=3.0, subtitle=None,
           captions=None, credits=None, credits_duration=8.0,
           bg_blur=30, keep_temp=False):
    """主流程：片头 + 修复图(语音) [+ 片尾出处卡] → 合成 mp4。

    参数：
      opening     开端照片路径
      image       最终修复图路径
      audio       TTS 语音 wav 路径
      out         输出 mp4 路径
      subtitle    底部固定单行字幕（标签/标题）
      captions    讲解词全文，自动切句生成随语音显示的动态字幕
      credits     片尾出处卡多行文本（list 或换行分隔 str），首行为标题
    """
    if isinstance(credits, str):
        credits = credits.splitlines()
    ffmpeg = find_ffmpeg()
    # 统一转绝对路径：主体段 ffmpeg 会在临时目录下运行，相对路径会失效
    opening = Path(opening).resolve()
    image = Path(image).resolve()
    audio = Path(audio).resolve()
    out = Path(out).resolve()
    audio_dur = wav_duration(audio)

    tmpdir = Path(tempfile.mkdtemp(prefix="video_"))
    segs = []
    try:
        # 字幕字体：复制到临时目录后用相对名引用，规避盘符冒号转义问题
        fontfile = None
        if subtitle or captions:
            font_path = find_cjk_font()
            if font_path is None:
                raise SystemExit("[!] 未找到中文字体，无法烧录字幕（可先去掉 --subtitle/--captions 跑通）")
            shutil.copy(Path(font_path), tmpdir / "font.ttf")
            fontfile = "font.ttf"

        # 1) 片头段（开端照片，静音）
        op_cover = fit_cover(opening, tmpdir / "opening_cover.png",
                             size=DEFAULT_SIZE, bg_blur=bg_blur)
        op_seg = tmpdir / "seg_00_opening.mp4"
        _segment(ffmpeg, op_cover, opening_duration, audio=None, out=op_seg)
        segs.append(op_seg)

        # 2) 主体段（修复图 + 语音 + 可选字幕，时长 = 音频时长）
        body_cover = fit_cover(image, tmpdir / "body_cover.png",
                               size=DEFAULT_SIZE, bg_blur=bg_blur)
        body_seg = tmpdir / "seg_01_body.mp4"
        _segment(ffmpeg, body_cover, audio_dur, audio=audio,
                 out=body_seg, subtitle=subtitle, captions=captions,
                 fontfile=fontfile, cwd=str(tmpdir))
        segs.append(body_seg)

        # 2.5) 片尾出处卡（可选：史实出处 + AIGC 来源标注）
        if credits:
            font_path = find_cjk_font()
            if font_path is None:
                raise SystemExit("[!] 未找到中文字体，无法生成片尾卡")
            cred_img = _make_credits_image(credits, font_path, DEFAULT_SIZE,
                                           tmpdir / "credits.png")
            cred_seg = tmpdir / "seg_02_credits.mp4"
            _segment(ffmpeg, cred_img, credits_duration, audio=None, out=cred_seg)
            segs.append(cred_seg)

        # 3) 拼接
        _concat(ffmpeg, segs, out)
        print(f"[ok] 已生成视频 → {Path(out).resolve()}")
        return out
    finally:
        if not keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="阶段5.4 视频合成：片头 + 修复图 + 语音")
    ap.add_argument("--opening", type=str, default=str(DEFAULT_OPENING),
                    help="开端照片路径（默认 guangying/opening.jpg）")
    ap.add_argument("--image", type=str, required=True, help="修复图路径")
    ap.add_argument("--audio", type=str, required=True, help="语音 wav 路径")
    ap.add_argument("--out", type=str, default=str(ROOT.parent / "out" / "video.mp4"),
                    help="输出 mp4 路径")
    ap.add_argument("--opening-duration", type=float, default=3.0, help="片头秒数")
    ap.add_argument("--subtitle", type=str, default=None, help="主体段底部固定单行字幕（标签/标题）")
    ap.add_argument("--captions", type=str, default=None, help="讲解词全文，自动切句生成随语音的逐句动态字幕")
    ap.add_argument("--credits-file", type=str, default=None, help="片尾出处卡多行文本文件（首行标题，可选）")
    ap.add_argument("--credits-duration", type=float, default=8.0, help="片尾卡时长（秒）")
    ap.add_argument("--bg-blur", type=int, default=30, help="模糊背景强度")
    ap.add_argument("--keep-temp", action="store_true", help="保留临时文件便于调试")
    args = ap.parse_args()

    credits = None
    if args.credits_file:
        credits = Path(args.credits_file).read_text(encoding="utf-8").splitlines()

    render(args.opening, args.image, args.audio, args.out,
           opening_duration=args.opening_duration, subtitle=args.subtitle,
           captions=args.captions, credits=credits,
           credits_duration=args.credits_duration,
           bg_blur=args.bg_blur, keep_temp=args.keep_temp)


if __name__ == "__main__":
    main()