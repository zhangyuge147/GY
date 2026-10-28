# -*- coding: utf-8 -*-
"""补抓第二批候选（杨贵生平、焦裕禄生平相关）"""
import pathlib
import requests

RAW = pathlib.Path(__file__).resolve().parent
OUT = RAW / "fetched5"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

SRC = [
    ("杨贵", "D5", "开凿“人工天河”红旗渠的历史启迪（人民网党史 2022）", "http://dangshi.people.com.cn/n1/2022/0129/c436975-32342981.html"),
    ("杨贵", "D6", "鉴往知来，跟着总书记学历史丨“人工天河”镌刻下了什么（人民网时政 2022）", "http://politics.people.com.cn/n1/2022/1030/c1001-32554884.html"),
    ("杨贵", "D7", "红旗渠精神历久弥新永不过时（人民网党史 2021）", "http://dangshi.people.com.cn/n1/2021/1116/c436975-32283381.html"),
    ("焦裕禄", "E4", "穿越一个甲子的精神回响——写在焦裕禄逝世60周年之际（人民网时政 2024）", "http://politics.people.com.cn/n1/2024/0514/c1001-40235869.html"),
    ("焦裕禄", "E5", "焦裕禄同志逝世60周年纪念活动举行（人民网河南 2024）", "http://henan.people.com.cn/n2/2024/0515/c351638-40844032.html"),
]


def clean(html):
    import re
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
    tried = [url] + (["http://" + url[8:]] if url.startswith("https://") else [])
    ok = False
    for u in tried:
        try:
            r = requests.get(u, headers=UA, timeout=25)
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding or "utf-8"
            if r.status_code != 200:
                print("   status", r.status_code, u); continue
            txt = clean(r.text)
            if len(txt) < 200:
                print("   short", len(txt), u); continue
            (OUT / f"{name}_{label}.txt").write_text(
                f"URL: {u}\n标题: {title}\n{'='*60}\n{txt[:14000]}", encoding="utf-8")
            print(f"[OK] {name}_{label} {len(txt)} chars")
            ok = True
            break
        except Exception as e:
            print("   ", type(e).__name__, u[:80])
    if not ok:
        print(f"[FAIL] {name}_{label}")
