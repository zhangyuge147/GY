# -*- coding: utf-8 -*-
"""抓取权威源正文，供人工逐条核对（仅临时校验用，不进入知识库）"""
import json, re, pathlib, sys

RAW = pathlib.Path(__file__).resolve().parent
OUT = RAW / "fetched"
OUT.mkdir(exist_ok=True)

data = json.loads((RAW / "authority_sources.json").read_text(encoding="utf-8"))

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}


def clean(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    html = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    html = re.sub(r"[ \t\u3000]+", " ", html)
    html = re.sub(r"\n\s*\n+", "\n", html)
    return html.strip()


ok = 0
for p in data["人物"]:
    name = p["姓名"]
    for i, s in enumerate(p.get("权威源", []), 1):
        url = s["url"]
        try:
            import requests
            r = requests.get(url, headers=UA, timeout=20, allow_redirects=True)
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding or "utf-8"
            txt = clean(r.text)
            if len(txt) < 200:
                print(f"[SHORT] {name} #{i} len={len(txt)} {url}")
                continue
            f = OUT / f"{name}_{i}.txt"
            f.write_text(f"URL: {url}\n标题: {s.get('标题','')}\n来源: {s.get('来源','')}\n"
                         f"{'='*60}\n{txt[:12000]}", encoding="utf-8")
            ok += 1
            print(f"[OK] {name} #{i} -> {f.name} ({len(txt)} chars)")
        except Exception as e:
            print(f"[FAIL] {name} #{i} {type(e).__name__}: {e}")

print(f"\n完成：成功 {ok} 篇 -> {OUT}")
