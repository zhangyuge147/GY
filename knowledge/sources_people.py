# -*- coding: utf-8 -*-
"""
第二批人物史料采集脚本
====================
作用（对齐 collect_hero.py 的硬约束，不改动其逻辑）：
  1. 用人民网站内检索接口，按姓名检索权威原文（人民网/中国共产党新闻网/党史频道等）；
  2. 只保留白名单域名、标题含姓名的文章，打分排序取前若干篇；
  3. 抓取正文落盘 raw/fetched2/，并把精简事实摘录写到 raw/digest/（供人工撰写摘要时核对）；
  4. 把选中的出处（标题/URL/来源/年份）合并写入 raw/authority_sources.json。

用法（在 knowledge/ 目录下）：
  python sources_people.py             # 处理名单中所有"待核验"人物
  python sources_people.py 李大钊 方志敏  # 只处理指定人物
"""
import csv, json, re, sys, time
import pathlib
import requests

ROOT = pathlib.Path(__file__).resolve().parent
LIST_CSV = ROOT / "人物名单.csv"
RAW_JSON = ROOT / "raw" / "authority_sources.json"
FETCH_DIR = ROOT / "raw" / "fetched2"
DIGEST_DIR = ROOT / "raw" / "digest"

API = "http://search.people.cn/search-platform/front/search"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Content-Type": "application/json", "Referer": "http://search.people.cn/s/"}

WHITELIST = ("people.com.cn", "dswxyjy.org.cn", "12371.cn", "chinamartyrs.gov.cn",
             "gov.cn", "xinhuanet.com", "news.cn", "mod.gov.cn", "81.cn",
             "gmw.cn", "cctv.com")
GOOD_KW = ("生平", "简介", "英雄烈士谱", "数风流人物", "为了民族复兴", "永远的丰碑",
           "故事", "事迹", "家书", "遗书", "就义", "殉国", "忠魂", "铁骨", "铁肩",
           "初心", "创建", "战斗", "永生", "光明", "信仰", "赤诚", "将军", "碑文")
BAD_KW = ("活动", "展览", "开幕", "举行", "举办", "开展", "座谈", "研讨", "揭牌",
          "授牌", "首发", "亮相", "发布会", "评选", "名单", "视频", "图集", "直播",
          "广告", "招聘", "客户端", "周年", "联展", "巡展", "特展", "征文", "故居",
          "遗迹", "旧址", "纪念馆", "陵园", "小学", "中学", "大学", "学院", "班级",
          "基金", "音乐剧", "思政课", "公演", "读书会", "分享会")
# 检索补充词：有些人物在人民网的常用称呼与本名不同
HINTS = {"江竹筠": ["江姐"], "叶挺": ["叶挺将军"], "瞿秋白": ["秋白"]}
GOOD_HOST = ("dangshi", "cpc", "dangjian", "theory", "paper", "politics",
             "military", "history", "henan", "jl", "hb", "sc")


def is_authority(url):
    host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0].lower()
    return any(host == d or host.endswith("." + d) for d in WHITELIST)


def search(key, pages=2):
    recs, seen = [], set()
    for st in (1, None):
        got_any = False
        for pg in range(1, pages + 1):
            try:
                r = requests.post(API, json={"key": key, "page": pg, "limit": 30,
                                             "sortType": 1, "searchType": st},
                                  headers=UA, timeout=25)
                got = r.json().get("data", {}).get("records", []) or []
            except Exception as e:
                print(f"   [search] {type(e).__name__}，跳过该页")
                got = []
            if not got:
                break
            got_any = True
            for rec in got:
                u = rec.get("url", "")
                if not u or u in seen:
                    continue
                seen.add(u)
                recs.append(rec)
        if got_any:
            break
    return recs


def score(rec, name, aliases=()):
    t = re.sub(r"<[^>]+>", "", rec.get("title", ""))
    u = rec.get("url", "")
    names = (name,) + tuple(aliases)
    s = 0
    if any(n in t for n in names):
        s += 100
    if any(h in u for h in GOOD_HOST):
        s += 8
    s += sum(10 for k in GOOD_KW if k in t)
    s -= 40 * sum(1 for k in BAD_KW if k in t)
    if re.match(rf"^{name}\s*[：:—\-]", t):
        s += 40
    if re.search(r"/VIDE|/video|\.mp4", u):
        s -= 60
    return s, t


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


def get_text(url):
    for u in [url] + (["http://" + url[8:]] if url.startswith("https://") else []):
        for _ in range(2):
            try:
                r = requests.get(u, headers={"User-Agent": UA["User-Agent"]}, timeout=25)
                if not r.encoding or r.encoding.lower() == "iso-8859-1":
                    r.encoding = r.apparent_encoding or "utf-8"
                if r.status_code == 200:
                    t = clean(r.text)
                    if len(t) > 300:
                        return u, t
                break
            except Exception:
                time.sleep(1)
    return None, ""


def digest(text, name, aliases=(), cap=900):
    """抽取含姓名或年份的句子，压成可核对的事实摘录。"""
    names = (name,) + tuple(aliases)
    sents = re.split(r"(?<=[。！？])", text)
    out, seen = [], set()
    for s in sents:
        s = s.strip()
        if len(s) < 8 or len(s) > 220:
            continue
        if any(n in s for n in names) or re.search(r"1[89]\d{2}年", s) or "牺牲" in s or "就义" in s:
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            if sum(len(x) for x in out) >= cap:
                break
    return "".join(out)


def year_of(t):
    """优先从 URL 的 /YYYY/ 取年份，其次取文本中第一个年份。"""
    m = re.search(r"/(20\d{2}|19\d{2})/\d{2}", t or "")
    if m:
        return m.group(1)
    m = re.search(r"(1[89]\d{2}|20\d{2})", t or "")
    return m.group(1) if m else ""


def main():
    targets = sys.argv[1:]
    rows = list(csv.DictReader(LIST_CSV.open(encoding="utf-8-sig")))
    if not targets:
        targets = [r["姓名"] for r in rows if r.get("摘要状态", "").strip() == "待核验"]
    meta = {r["姓名"]: r for r in rows}
    data = json.loads(RAW_JSON.read_text(encoding="utf-8")) if RAW_JSON.exists() else {"人物": []}
    by_name = {p["姓名"]: p for p in data.get("人物", [])}
    FETCH_DIR.mkdir(parents=True, exist_ok=True)
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)

    for name in targets:
        print(f"\n===== {name} =====")
        aliases = HINTS.get(name, [])
        recs, seen_url = [], set()
        for key in [name] + aliases:
            for rec in search(key, pages=4):
                u = rec.get("url", "")
                if u and u not in seen_url:
                    seen_url.add(u)
                    recs.append(rec)
        scored = sorted(((score(r, name, aliases)[0], score(r, name, aliases)[1], r.get("url", ""))
                         for r in recs), key=lambda x: -x[0])
        cands, seen_title = [], set()
        for sc, title, url in scored:
            if sc < 100 or not is_authority(url):
                continue
            key = re.sub(r"\s+", "", title)[:36]
            if key in seen_title:
                continue
            seen_title.add(key)
            cands.append((sc, title, url))
        if not cands:
            print("   [!] 无可用的白名单候选")
            continue

        chosen, digest_parts = [], []
        for sc, title, url in cands[:10]:
            if len(chosen) >= 4:
                break
            real, text = get_text(url)
            if not text or not any(n in text for n in (name,) + tuple(aliases)):
                continue
            (FETCH_DIR / f"{name}_{len(chosen)+1}.txt").write_text(
                f"URL: {real}\n标题: {title}\n{'='*60}\n{text[:14000]}", encoding="utf-8")
            chosen.append({"标题": title, "url": real,
                           "来源": real.split("/")[2], "年份": year_of(real)})
            digest_parts.append(f"\n--- 出处：{title}\nURL: {real}\n{digest(text, name, aliases)}")
            print(f"   [OK] {title[:60]} | {real[:100]}")

        if not chosen:
            print("   [!] 候选正文均抓取失败")
            continue

        (DIGEST_DIR / f"{name}.txt").write_text(
            f"# {name} 事实摘录（自动抽取，仅核对用）\n" + "".join(digest_parts), encoding="utf-8")

        person = by_name.get(name)
        if not person:
            person = {"姓名": name, "生卒": meta.get(name, {}).get("生卒", ""),
                      "身份": meta.get(name, {}).get("身份", ""), "权威源": []}
            data["人物"].append(person)
            by_name[name] = person
        exist = {s["url"] for s in person["权威源"]}
        for c in chosen:
            if c["url"] not in exist:
                person["权威源"].append(c)
        RAW_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   [saved] {len(chosen)} 条出处，摘录 -> raw/digest/{name}.txt")

    print("\n[done] 出处清单：", RAW_JSON)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
