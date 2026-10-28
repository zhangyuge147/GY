# -*- coding: utf-8 -*-
"""
受限联网检索（博查 Bocha Web Search API）：仅供「素材库检索不到 / 相似度过低」时兜底。

合规红线（项目 memory）：
  - 结果仅来自白名单权威源（党史 / 政府 / 权威党媒），并在前端标注「联网检索 · 未经人工核验」。
  - 仅作「未核验」补充参考，不进入库内出处体系（citations 保持空，走独立的 web_sources 字段）。

API key 读取：优先 BOCHA_API_KEY，compat 回退旧的 BING_SEARCH_KEY 字段名。
"""
import os

BOCHA_ENDPOINT = "https://api.bochaai.com/v1/web-search"

# 权威源白名单（按域名后缀匹配，含子域）
AUTHORITY_DOMAINS = (
    "people.com.cn",   # 人民网（含党史 / 时政子域）
    "12371.cn",        # 共产党员网
    "gov.cn",          # 各级政府网站
    "xinhuanet.com",   # 新华网
)


def _is_authority(url: str) -> bool:
    if not url:
        return False
    host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0].lower()
    return any(host == d or host.endswith("." + d) for d in AUTHORITY_DOMAINS)


def web_search(query: str, count: int = 5) -> list:
    """博查联网检索，返回 [{title, url, snippet}]（仅保留权威源）；失败返回 []。"""
    key = (os.environ.get("BOCHA_API_KEY", "").strip()
           or os.environ.get("BING_SEARCH_KEY", "").strip())
    if not key:
        return []
    try:
        import requests
    except ImportError:
        return []

    try:
        resp = requests.post(
            BOCHA_ENDPOINT,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"query": query, "freshness": "noLimit",
                  "summary": False, "count": max(5, count * 2)},
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        print(f"[search] 博查请求失败：{e}", flush=True)
        return []

    if resp.status_code != 200:
        print(f"[search] 博查返回 {resp.status_code}：{resp.text[:200]}", flush=True)
        return []

    try:
        value = resp.json().get("data", {}).get("webPages", {}).get("value", [])
    except (ValueError, AttributeError):
        return []

    out = []
    for p in value:
        url = p.get("url", "")
        if not _is_authority(url):
            continue
        out.append({"title": p.get("name", ""), "url": url,
                    "snippet": p.get("snippet", "")})
        if len(out) >= count:
            break
    return out