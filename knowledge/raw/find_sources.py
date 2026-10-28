# -*- coding: utf-8 -*-
"""用 DuckDuckGo HTML 版检索存量 5 人的权威出处链接（仅临时检索用，不进入知识库）。
结果只做候选，仍需人工打开原文核对。"""
import json, re, time, urllib.parse, html as H
import pathlib
import requests

RAW = pathlib.Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

WHITELIST = ("people.com.cn", "dswxyjy.org.cn", "12371.cn", "chinamartyrs.gov.cn",
             "gov.cn", "xinhuanet.com", "news.cn", "mod.gov.cn", "81.cn",
             "gmw.cn", "cctv.com")

QUERIES = {
    "杨靖宇": [
        "杨靖宇 奋勇向前 不畏牺牲的冲锋者 人民网",
        "杨靖宇 东三省著名的抗日民族英雄 党史",
        "爷爷杨靖宇的抗日传奇 人民网",
        "习近平讲述的故事 铁血将军杨靖宇",
    ],
    "吉鸿昌": [
        "吉鸿昌 我为抗日而死 数风流人物",
        "吉鸿昌 国破尚如此 我何惜此头",
        "吉鸿昌 旧军阀 共产主义战士 党史",
        "吉鸿昌 就义 12371 北京党史",
    ],
    "彭雪枫": [
        "彭雪枫 生平简介 人民网 党史",
        "乌云之中见青天 竹沟就是小延安",
        "虎胆英雄 彭雪枫 37岁 12371",
        "彭雪枫 战死疆场也心甘情愿",
    ],
    "杨贵": [
        "红旗渠总设计师杨贵 学习时报",
        "太行天河 红旗渠修建的前前后后",
        "从林县走出的四位县委书记",
        "铺席 鞋子先给民工用 杨贵",
    ],
    "焦裕禄": [
        "焦裕禄 双百人物 共产党员网",
        "一百堂党史课 焦裕禄精神",
        "永远不会过时的人民公仆精神 焦裕禄",
        "焦裕禄 兰考 县委书记的榜样 党史",
    ],
}


def ddg(q: str):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
    for _ in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.encoding = "utf-8"
            if r.status_code == 200 and "result__a" in r.text:
                return r.text
        except Exception as e:
            print("   retry", type(e).__name__)
        time.sleep(3)
    return ""


def parse(t: str):
    out = []
    for href, title in re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', t, re.S):
        href = H.unescape(href)
        m = re.search(r"uddg=([^&]+)", href)
        real = urllib.parse.unquote(m.group(1)) if m else href
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", H.unescape(title))).strip()
        out.append({"标题": title, "url": real})
    return out


def is_auth(u: str):
    return any(d in u for d in WHITELIST)


found = {}
for name, qs in QUERIES.items():
    print(f"\n===== {name} =====")
    seen = set()
    items = []
    for q in qs:
        print(f"- 检索: {q}")
        for it in parse(ddg(q)):
            if not is_auth(it["url"]) or it["url"] in seen:
                continue
            seen.add(it["url"])
            items.append(it)
            print("   [权威]", it["标题"][:70], "|", it["url"][:110])
        time.sleep(1.5)
    found[name] = items

(RAW / "_found.json").write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n候选已写入 raw/_found.json")
