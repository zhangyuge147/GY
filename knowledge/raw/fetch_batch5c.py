# -*- coding: utf-8 -*-
"""补抓焦裕禄生平候选页"""
import pathlib, re
import requests

RAW = pathlib.Path(__file__).resolve().parent
OUT = RAW / "fetched5"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

SRC = [
    ("焦裕禄", "E6", "焦裕禄和他的母亲（人民网党史 2021）", "http://dangshi.people.com.cn/n1/2021/1224/c436975-32315814.html"),
    ("焦裕禄", "E7", "树立和践行正确政绩观的一面旗帜（中国共产党新闻网 2026）", "http://cpc.people.com.cn/n1/2026/0720/c443712-40764010.html"),
    ("焦裕禄", "E8", "躲暗杀、斗匪首，揭秘不为人知的“强悍焦裕禄”（人民网党建 2022）", "http://dangjian.people.com.cn/n1/2022/0816/c117092-32503562.html"),
]


def clean(html):
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    html = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = (html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
                .replace("&lt;", "<").replace("&gt;", ">"))
    html = re.sub(r"[ \t\u3000]+", " ", html)
    html = re.sub(r"\n\s*\n+", "\n", html)
    return html.strip()


for name, label, title, url in SRC:
    try:
        r = requests.get(url, headers=UA, timeout=25)
        r.encoding = r.apparent_encoding if not r.encoding or r.encoding.lower() == "iso-8859-1" else r.encoding
        txt = clean(r.text)
        if r.status_code != 200 or len(txt) < 200:
            print(f"[FAIL] {label} {r.status_code} {len(txt)}"); continue
        (OUT / f"{name}_{label}.txt").write_text(
            f"URL: {url}\n标题: {title}\n{'='*60}\n{txt[:14000]}", encoding="utf-8")
        print(f"[OK] {name}_{label} {len(txt)} chars")
    except Exception as e:
        print(f"[FAIL] {label} {type(e).__name__}")
