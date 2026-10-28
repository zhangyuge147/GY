# -*- coding: utf-8 -*-
"""抓取存量 5 人候选权威源正文（临时核验用，落 raw/fetched5/，不入库）"""
import re, pathlib
import requests

RAW = pathlib.Path(__file__).resolve().parent
OUT = RAW / "fetched5"
OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

SRC = {
    "杨靖宇": [
        ("A1", "爷爷杨靖宇的抗日传奇（人民网党建频道 2015）", "http://dangjian.people.com.cn/n/2015/0813/c117092-27456047.html"),
        ("A2", "杨靖宇：东三省著名的抗日民族英雄（人民网党史频道 2015）", "http://dangshi.people.com.cn/n/2015/0829/c85037-27530434.html"),
        ("A3", "习近平讲述的故事丨铁血将军杨靖宇（人民网党史 2021）", "http://dangshi.people.com.cn/n1/2021/0608/c436975-32125395.html"),
        ("A4", "杨靖宇：奋勇向前、不畏牺牲的冲锋者（人民日报 2025）", "http://jl.people.com.cn/n2/2025/0622/c349771-41267568.html"),
        ("A5", "杨靖宇：奋勇向前、不畏牺牲的冲锋者（人民日报电子版 2025）", "https://paper.people.com.cn/rmrb/pc/content/202506/22/content_30081152.html"),
    ],
    "吉鸿昌": [
        ("B1", "吉鸿昌：我为抗日而死，死得光明正大（中国共产党新闻网 2025）", "http://cpc.people.com.cn/n1/2025/0624/c443712-40507340.html"),
        ("B2", "吉鸿昌：国破尚如此，我何惜此头（中国共产党新闻网 2025）", "http://cpc.people.com.cn/n1/2025/1026/c443712-40589495.html"),
        ("B3", "吉鸿昌（数风流人物，新华网 2021）", "http://www.xinhuanet.com/politics/2021-05/17/c_1127456670.htm"),
        ("B4", "吉鸿昌（数风流人物，中国军网 2021）", "http://www.81.cn/big5/yw/2021-05/17/content_10036078.htm"),
        ("B5", "吉鸿昌（共产党员网 2025）", "https://www.12371.cn/2025/07/01/VIDE1751339102870325.shtml"),
    ],
    "彭雪枫": [
        ("C1", "乌云之中见青天，竹沟就是小延安（中国共产党新闻网 2023）", "http://cpc.people.com.cn/n1/2023/0923/c443712-40083709.html"),
        ("C2", "彭雪枫：勤俭无私 赤诚报国（中国共产党新闻网 2023）", "http://cpc.people.com.cn/n1/2023/1115/c443712-40118499.html"),
        ("C3", "铭记历史 缅怀先烈｜拂晓星芒成霞光（人民网军事 2025）", "http://military.people.com.cn/n1/2025/0629/c1011-40510929.html"),
    ],
    "杨贵": [
        ("D1", "红旗渠总设计师杨贵（中国共产党新闻网 2023，来源学习时报）", "http://cpc.people.com.cn/n1/2023/0106/c443712-32600998.html"),
        ("D2", "走近红旗渠（人民网河南频道 2023）", "http://henan.people.com.cn/n2/2023/0109/c351638-40258784.html"),
        ("D3", "从红旗渠精神中汲取奋进力量（人民网党建频道 2024）", "http://dangjian.people.com.cn/n1/2024/0723/c117092-40283108.html"),
        ("D4", "深入把握红旗渠精神内核（中国共产党新闻网 2022）", "http://cpc.people.com.cn/n1/2022/1205/c443712-32580529.html"),
    ],
    "焦裕禄": [
        ("E1", "焦裕禄精神永远不会过时（人民网党建频道 2022）", "http://dangjian.people.com.cn/n1/2022/0817/c117092-32504423.html"),
        ("E2", "走进焦裕禄的精神世界（中国共产党新闻网 2023）", "http://cpc.people.com.cn/n1/2023/0522/c443712-32691352.html"),
        ("E3", "焦裕禄精神（人民网理论频道 2021）", "http://theory.people.com.cn/n1/2021/1116/c40531-32283159.html"),
    ],
}


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


def get(url):
    tried = [url]
    if url.startswith("https://"):
        tried.append("http://" + url[len("https://"):])
    for u in tried:
        try:
            r = requests.get(u, headers=UA, timeout=25, allow_redirects=True)
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding or "utf-8"
            if r.status_code == 200:
                return u, clean(r.text)
            print(f"    status {r.status_code} {u}")
        except Exception as e:
            print(f"    {type(e).__name__} {u}")
    return None, ""


for name, items in SRC.items():
    print(f"\n== {name}")
    for label, title, url in items:
        real, txt = get(url)
        if len(txt) < 200:
            print(f"   [FAIL] {label} {url}")
            continue
        (OUT / f"{name}_{label}.txt").write_text(
            f"URL: {real}\n标题: {title}\n{'='*60}\n{txt[:14000]}", encoding="utf-8")
        print(f"   [OK] {label} {len(txt)} chars -> {name}_{label}.txt")
