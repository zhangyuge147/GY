# -*- coding: utf-8 -*-
"""
红色人物史料采集流水线（小样 10 人）：权威源直采 / 合规校验 / 待核验落盘
======================================================================
背景与硬约束（对齐 knowledge/rag.py 与项目 memory）：
  1. 史料知识库只存"结构化事实摘要 + 逐条标注权威出处"，不搬原文，严禁编造史实。
  2. 仅采信白名单权威源（党史 / 政府 / 权威党媒），非白名单一律拒绝入库。
  3. 采集结果一律先落"待核验区" corpus_pending/，人工复核通过后才允许进 corpus/。
  4. 本机网络对部分权威站（新华网 / 共产党员网等）可能不可达，check 只做可达性提示，
     不影响其作为"出处"保留（需人工在浏览器复核）。

数据源：
  人物名单.csv                     —— 人物元数据与摘要状态（已核验 / 待核验）
  raw/authority_sources.json       —— 每人权威源（标题 / URL / 来源 / 年份），由检索步骤产出

用法（在 guangying/ 目录运行）：
  python knowledge/collect_hero.py --check        # 校验权威源 URL（白名单 + 可达性）
  python knowledge/collect_hero.py --draft        # 为"待核验"人物生成摘要骨架到 corpus_pending/
  python knowledge/collect_hero.py --validate     # 用 build_index 解析器校验待核验摘要格式
  python knowledge/collect_hero.py --search       # 博查 API 批量补权威源（网络不通自动跳过）
  参数可组合：--check --validate
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # knowledge/
LIST_CSV = ROOT / "人物名单.csv"
RAW_JSON = ROOT / "raw" / "authority_sources.json"
PENDING = ROOT / "corpus_pending"
CORPUS = ROOT / "corpus"
REPORT = ROOT / "raw" / "采集校验报告.txt"
ENV_FILE = ROOT.parent / ".env"

# 权威源白名单（域名后缀匹配，含子域）
AUTHORITY_DOMAINS = (
    "people.com.cn",      # 人民网 / 中国共产党新闻网 cpc / 党史频道 dangshi
    "dswxyjy.org.cn",     # 中央党史和文献研究院
    "12371.cn",           # 共产党员网
    "gov.cn",             # 各级政府网站（含中华英烈网 chinamartyrs.gov.cn）
    "xinhuanet.com",      # 新华网
    "news.cn",            # 新华网新域名
    "mod.gov.cn",         # 国防部
    "81.cn",              # 中国军网
    "gmw.cn",             # 光明日报
    "cctv.com",           # 央视网
)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"


def is_authority(url: str) -> bool:
    host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0].lower()
    return any(host == d or host.endswith("." + d) for d in AUTHORITY_DOMAINS)


def load_env(path: Path = ENV_FILE):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        import os
        os.environ.setdefault(k.strip(), v.strip())


def load_list():
    with LIST_CSV.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_raw():
    if not RAW_JSON.exists():
        return {"人物": []}
    return json.loads(RAW_JSON.read_text(encoding="utf-8"))


def save_raw(data):
    RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    RAW_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def reachable(url: str, timeout: int = 10) -> str:
    try:
        import requests
    except ImportError:
        return "NO_REQUESTS(请 pip install requests)"
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True,
                         headers={"User-Agent": UA})
        return f"OK {r.status_code}"
    except Exception as e:                     # 网络抖动 / SSL / 超时都归为不可达
        return f"UNREACHABLE({type(e).__name__})"


def cmd_check():
    """校验权威源：白名单 + 本机可达性，写报告文件。"""
    data = load_raw()
    lines = ["红色人物史料采集 · 权威源校验报告", "=" * 46,
             "说明：UNREACHABLE 仅代表本机网络不可达，不代表来源无效；"
             "需人工在浏览器复核。", ""]
    ok = bad_domain = unreach = 0
    for p in data.get("人物", []):
        lines.append(f"【{p['姓名']}】（{p.get('生卒','')}）")
        for s in p.get("权威源", []):
            url = s.get("url", "")
            if not is_authority(url):
                bad_domain += 1
                lines.append(f"  ✗ 非白名单：{url}")
                continue
            st = reachable(url)
            if st.startswith("OK"):
                ok += 1
            else:
                unreach += 1
            lines.append(f"  - {st:22s} {s.get('来源','')}《{s.get('标题','')}》")
            lines.append(f"    {url}")
        lines.append("")
    lines.append(f"合计：白名单通过可达 {ok} 条 | 本机不可达 {unreach} 条 | 非白名单 {bad_domain} 条")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(["=" * 46] + [l for l in lines if l and not l.startswith("=")]))
    print(f"\n[report] 已写入 {REPORT}")
    return bad_domain == 0


def _skeleton(name, meta, sources):
    cite = []
    for i, s in enumerate(sources, 1):
        src = s.get("来源", "")
        title = s.get("标题", "")
        year = (s.get("年份") or "").strip()
        yr = f"，{year}年" if year else ""
        cite.append(f"{i}. {src}《{title}》（{s.get('url','')}{yr}）")
    return (
        f"# {name}人物摘要\n\n"
        f"## 基本信息\n（待人工补充：性别、民族、原名/化名、籍贯、生卒、身份与荣誉）\n\n"
        f"## 生平主要经历\n- 待人工按时间顺序补充（每条须能在下方引用出处中回源）\n\n"
        f"## 核心事迹：待补充\n- 待人工补充\n\n"
        f"## 精神内涵\n（待人工补充，用法克制、符合党史权威表述）\n\n"
        f"## 引用出处\n" + "\n".join(cite) + "\n"
    )


def cmd_draft():
    """为"待核验"人物生成摘要骨架（已存在则跳过，不覆盖人工撰写稿）。"""
    rows = load_list()
    raw = {p["姓名"]: p for p in load_raw().get("人物", [])}
    PENDING.mkdir(parents=True, exist_ok=True)
    made = skipped = 0
    for row in rows:
        name = row["姓名"].strip()
        if row.get("摘要状态", "").strip() != "待核验":
            continue
        out = PENDING / f"{name}.txt"
        if out.exists():
            skipped += 1
            print(f"[skip] {name} 已存在，保留人工稿")
            continue
        sources = raw.get(name, {}).get("权威源", [])
        out.write_text(_skeleton(name, row, sources), encoding="utf-8")
        made += 1
        print(f"[draft] {name} -> {out.name}（{len(sources)} 条出处）")
    print(f"[done] 新建 {made} 份骨架，跳过 {skipped} 份")


def cmd_validate():
    """用 build_index 的解析器校验待核验摘要：能否解析、chunk 数、出处数。"""
    sys.path.insert(0, str(ROOT))
    from build_index import parse_summary, chunkize   # 复用正式索引的解析规则

    files = sorted(PENDING.glob("*.txt"))
    if not files:
        print("[!] corpus_pending/ 无摘要文件")
        return False
    all_ok = True
    for f in files:
        parsed = parse_summary(f.read_text(encoding="utf-8"))
        cs = chunkize(parsed)
        n_src = len(parsed["sources"])
        flag = "OK"
        if not cs or n_src < 2 or parsed["person"] == "未命名人物":
            flag = "FAIL"
            all_ok = False
        print(f"[{flag}] {f.name}: person={parsed['person']} "
              f"chunks={len(cs)} 出处={n_src}")
        for s in parsed["sources"][:3]:
            print(f"        - {s[:70]}")
    print(f"\n[done] 共 {len(files)} 份，校验{'全部通过' if all_ok else '存在 FAIL'}")
    return all_ok


def cmd_search():
    """博查 API 批量补权威源（本机网络不通时自动跳过，不影响已有 raw 数据）。"""
    load_env()
    sys.path.insert(0, str(ROOT))
    try:
        from search import web_search
    except ImportError:
        print("[!] 无法导入 search.py，跳过")
        return False
    rows = [r for r in load_list() if r.get("摘要状态", "").strip() == "待核验"]
    data = load_raw()
    by_name = {p["姓名"]: p for p in data.get("人物", [])}
    added = 0
    for row in rows:
        name = row["姓名"].strip()
        res = web_search(f"{name} 事迹 生平", count=5)
        if not res:
            print(f"[skip] {name}: 检索无结果（网络受限或 API 不可用）")
            continue
        person = by_name.get(name)
        if not person:
            person = {"姓名": name, "生卒": row.get("生卒", ""),
                      "身份": row.get("身份", ""), "权威源": []}
            data["人物"].append(person)
            by_name[name] = person
        exist = {s.get("url") for s in person["权威源"]}
        for r in res:
            if r["url"] in exist or not is_authority(r["url"]):
                continue
            person["权威源"].append({"标题": r["title"], "url": r["url"],
                                     "来源": "联网检索", "年份": ""})
            exist.add(r["url"])
            added += 1
        print(f"[search] {name}: 新增 {len(res)} 条候选")
    save_raw(data)
    print(f"[done] 新增权威源 {added} 条 → {RAW_JSON}")
    return added > 0


def main():
    ap = argparse.ArgumentParser(description="红色人物史料采集流水线")
    ap.add_argument("--check", action="store_true", help="校验权威源 URL")
    ap.add_argument("--draft", action="store_true", help="生成待核验摘要骨架")
    ap.add_argument("--validate", action="store_true", help="校验摘要格式")
    ap.add_argument("--search", action="store_true", help="联网补权威源")
    a = ap.parse_args()
    if not any([a.check, a.draft, a.validate, a.search]):
        ap.print_help()
        return 0
    ok = True
    if a.search:
        ok &= cmd_search()
    if a.draft:
        cmd_draft()
    if a.check:
        ok &= cmd_check()
    if a.validate:
        ok &= cmd_validate()
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
