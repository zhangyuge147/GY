# -*- coding: utf-8 -*-
"""
光影重生 · 盲评问卷生成器（任务 3.11）
========================================
三结合评测方案的第三环：人工盲评。从 3.10 修复集（28 张）按类别分层抽样
12 组（building 2 / rural 3 / scene 3 / portrait 4），生成单文件 HTML 交互
问卷：每组并列「原图参照 + A/B 两图」，A/B 中一张为修复前退化图、一张为
修复后结果，顺序随机（固定种子可复现）；评分员 1-5 分别给 A/B 打分，并
选择偏好（A 更好 / B 更好 / 差不多）。所有图片以 base64 缩略图（长边 600、
JPEG q85）内嵌，问卷单文件自包含、浏览器打开即可作答。

输出：
  blind_judge/questionnaire.html   交互问卷（分发给 3 名评分员）
评分回收：评分员作答后在页面填写编号并"下载评分 JSON"（judge_<编号>.json），
  回收 3 份后运行 aggregate_judges.py 汇总。

用法（在 guangying/ 目录、guangying 环境下）：
  python evaluation/blind_judge/make_questionnaire.py
  python evaluation/blind_judge/make_questionnaire.py --seed 20261017 --n-per 2,3,3,4
"""
import argparse
import base64
import io
import random
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent       # evaluation/
PROJECT = ROOT.parent                                # guangying/
BENCH = PROJECT / "restoration" / "data" / "bench"
DEGRADED = ROOT / "data" / "degraded"
REPAIRED = ROOT / "data" / "repaired"
OUT_DIR = ROOT / "blind_judge"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CATS = ("building", "rural", "scene", "portrait")
THUMB = 600            # 缩略图长边
JPEG_Q = 85


def imread_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return img


def thumb_b64(bgr: np.ndarray) -> str:
    """缩略图（长边 THUMB，JPEG q85）转 base64 data URL"""
    h, w = bgr.shape[:2]
    k = THUMB / max(h, w)
    if k < 1:
        bgr = cv2.resize(bgr, (round(w * k), round(h * k)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def category(name: str) -> str:
    for c in CATS:
        if name.startswith(c):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser(description="光影重生·盲评问卷生成（任务 3.11）")
    ap.add_argument("--seed", type=int, default=20261017,
                    help="抽样与 A/B 顺序随机种子（默认 20261017，可复现）")
    ap.add_argument("--n-per", type=str, default="2,3,3,4",
                    help="每类抽样数量（building,rural,scene,portrait，默认 2,3,3,4 = 12 组）")
    ap.add_argument("--scale", type=int, default=4, help="修复结果倍率（命名匹配）")
    args = ap.parse_args()

    n_per = [int(x) for x in args.n_per.split(",")]
    assert len(n_per) == len(CATS) and sum(n_per) > 0
    tag = f"_sr_x{args.scale}"
    rep_map = {p.stem[:-len(tag)]: p for p in REPAIRED.glob("*")
               if p.suffix.lower() in IMG_EXTS and p.name.endswith(f"{tag}.png")}

    # 分层抽样（固定种子可复现）
    rng = random.Random(args.seed)
    picked = []
    for cat, n in zip(CATS, n_per):
        cand = [p.stem for p in DEGRADED.glob("*")
                if p.suffix.lower() in IMG_EXTS and p.stem in rep_map
                and p.stem.startswith(cat)]
        if len(cand) < n:
            print(f"[!] {cat} 候选 {len(cand)} 不足 {n}，按实际抽样")
        picked.extend(rng.sample(sorted(cand), min(n, len(cand))))
    print(f"分层抽样 {len(picked)} 组：{picked}")

    # 生成问卷数据：每组 {id, gt, a:{data,isRepair}, b:{data,isRepair}}
    groups = []
    for stem in picked:
        gt_path = next((BENCH / (stem + ext) for ext in (".jpg", ".jpeg", ".png", ".bmp")
                        if (BENCH / (stem + ext)).exists()), None)
        if gt_path is None:
            print(f"[!] 跳过 {stem}：bench 无原图")
            continue
        gt_b64 = thumb_b64(imread_bgr(gt_path))
        base_b64 = thumb_b64(imread_bgr(DEGRADED / f"{stem}.png"))
        rep_b64 = thumb_b64(imread_bgr(rep_map[stem]))
        # A/B 顺序随机（修复后 isRepair 随机分给 A 或 B）
        if rng.random() < 0.5:
            a, b = rep_b64, base_b64
            a_rep, b_rep = True, False
        else:
            a, b = base_b64, rep_b64
            a_rep, b_rep = False, True
        groups.append({
            "id": stem,
            "cat": category(stem),
            "gt": gt_b64,
            "a": {"data": a, "repair": a_rep},
            "b": {"data": b, "repair": b_rep},
        })

    html = render_html(args.seed, groups)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "questionnaire.html"
    out.write_text(html, encoding="utf-8")
    print(f"问卷已生成：{out}（{len(groups)} 组，"
          f"{out.stat().st_size // 1024} KB）")
    print("分发方式：3 名评分员浏览器打开问卷作答 → 填编号 → 下载评分 JSON"
          "\n（judge_<编号>.json）→ 回收后运行 aggregate_judges.py 汇总")


def render_html(seed: int, groups: list) -> str:
    import json
    data_json = json.dumps(groups, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>光影重生 · 修复效果盲评问卷</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 1080px;
        margin: 0 auto; padding: 16px; background: #f5f6f8; color: #222; }}
  h1 {{ font-size: 22px; }} h2 {{ font-size: 16px; color: #444; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px; margin: 14px 0;
         box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card h3 {{ margin: 4px 0 8px; font-size: 15px; color: #333; }}
  .ref {{ text-align: center; margin-bottom: 10px; }}
  .ref img {{ max-width: 260px; border: 1px solid #ddd; border-radius: 6px; }}
  .pair {{ display: flex; gap: 12px; }}
  .pair .side {{ flex: 1; text-align: center; }}
  .pair img {{ width: 100%; border-radius: 6px; border: 1px solid #ccc; }}
  .score {{ display: flex; gap: 4px; justify-content: center; margin: 6px 0; }}
  .score label {{ cursor: pointer; padding: 4px 8px; border: 1px solid #bbb;
         border-radius: 5px; background: #fff; }}
  .score label.on {{ background: #2f6fdb; color: #fff; border-color: #2f6fdb; }}
  .pref {{ text-align: center; margin-top: 4px; font-size: 14px; }}
  .pref label {{ margin: 0 10px; cursor: pointer; }}
  .pref input {{ margin-right: 3px; }}
  footer {{ position: sticky; bottom: 0; background: #f5f6f8; padding: 10px 0;
          border-top: 1px solid #ddd; }}
  footer input {{ padding: 6px; width: 120px; }}
  footer button {{ padding: 7px 16px; background: #2f6fdb; color: #fff;
         border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }}
  .done {{ background: #e8f5e9; }}
  .tip {{ color: #777; font-size: 13px; }}
</style>
</head>
<body>
<h1>修复效果盲评问卷</h1>
<p class="tip">每张卡片展示一张<b>原图参照</b>（彩色原始照片）与 <b>A / B 两张图</b>——
其中一张是<b>修复前</b>的退化图，另一张是<b>修复后</b>的结果，顺序已打乱。
请仅凭观感回答：① 分别给 A、B 的<b>质量分</b>（综合自然度+清晰度+可信度，1-5 分）；
② 选一张你<b>整体观感更好</b>的图。评分进度自动保存在本浏览器。</p>
<p class="tip"><b>评分档位：</b>5=很好（清晰自然、无明显伪影，如同真实照片）
｜ 4=好（整体自然，仅轻微可辨缺陷）｜ 3=一般（可接受，但可见模糊/伪影/不自然色）
｜ 2=差（明显损坏：大片模糊、强伪影、颜色失真）｜ 1=很差（严重损坏，几乎无法辨认）。</p>
<div id="app"></div>
<footer>
  <input id="judge" placeholder="评分员编号（如 1/2/3）">
  <button onclick="downloadJson()">下载评分 JSON</button>
  <span class="tip" id="status"></span>
</footer>
<script>
const DATA = {data_json};
const N = DATA.length;
let ans = {{}};   // id -> {{scoreA, scoreB, pref:'A'|'B'|'tie'}}

function esc(s) {{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;'); }}

function build() {{
  const app = document.getElementById('app');
  DATA.forEach((g, i) => {{
    const card = document.createElement('div');
    card.className = 'card';
    card.id = 'c' + g.id;
    const scores = (g) => `
      <div class="score" data-for="scoreA">${{[1,2,3,4,5].map(v =>
        `<label data-v="${{v}}">${{v}}</label>`).join('')}}</div>
      <div class="score" data-for="scoreB">${{[1,2,3,4,5].map(v =>
        `<label data-v="${{v}}">${{v}}</label>`).join('')}}</div>`;
    card.innerHTML = `
      <h3>${{i+1}}. ${{esc(g.id)}} <span class="tip">（${{esc(g.cat)}}）</span></h3>
      <div class="ref"><img src="${{g.gt}}" alt="原图参照"></div>
      <div class="pair">
        <div class="side"><div class="tip">A 图</div>
          <img src="${{g.a.data}}" alt="A"><div class="tip">A 质量分</div>${{scores(g)}}</div>
        <div class="side"><div class="tip">B 图</div>
          <img src="${{g.b.data}}" alt="B"><div class="tip">B 质量分</div>${{scores(g)}}</div>
      </div>
      <div class="pref">整体更偏好：
        <label><input type="radio" name="p${{g.id}}" value="A">A 更好</label>
        <label><input type="radio" name="p${{g.id}}" value="B">B 更好</label>
        <label><input type="radio" name="p${{g.id}}" value="tie">差不多</label>
      </div>`;
    app.appendChild(card);
  }});
  // 评分点击交互
  document.querySelectorAll('.score').forEach(box => {{
    box.addEventListener('click', e => {{
      const lbl = e.target.closest('label'); if (!lbl) return;
      const forKey = box.dataset.for;        // 'scoreA' | 'scoreB'
      const g = DATA.find(x => box.closest('.card').id === 'c' + x.id);
      const v = Number(lbl.dataset.v);
      if (!ans[g.id]) ans[g.id] = {{scoreA:0, scoreB:0, pref:null}};
      ans[g.id][forKey === 'scoreA' ? 'scoreA' : 'scoreB'] = v;
      box.querySelectorAll('label').forEach(l =>
        l.classList.toggle('on', l.dataset.v === String(v)));
      save(); markDone(g.id);
    }});
  }});
  // 偏好交互
  document.querySelectorAll('.pref input').forEach(r => {{
    r.addEventListener('change', e => {{
      const g = DATA.find(x => x.id === r.name.slice(1));
      if (!ans[g.id]) ans[g.id] = {{scoreA:0, scoreB:0, pref:null}};
      ans[g.id].pref = r.value;
      save(); markDone(g.id);
    }});
  }});
}}

function markDone(id) {{
  const a = ans[id]; if (!a) return;
  if (a.scoreA > 0 && a.scoreB > 0 && a.pref) {{
    document.getElementById('c' + id).classList.add('done');
  }}
}}

function save() {{
  try {{ localStorage.setItem('blind_ans', JSON.stringify(ans)); }} catch(e) {{}}
  const doneN = Object.values(ans).filter(a =>
    a.scoreA > 0 && a.scoreB > 0 && a.pref).length;
  document.getElementById('status').textContent =
    `已完成 ${{doneN}} / ${{N}} 组`;
}}

function downloadJson() {{
  const j = document.getElementById('judge').value.trim() || 'x';
  const payload = {{
    judge: j, seed: {seed}, answers: ans,
    meta: DATA.map(g => ({{id: g.id, cat: g.cat,
      A_is_repair: g.a.repair, B_is_repair: g.b.repair}}))
  }};
  const blob = new Blob([JSON.stringify(payload, null, 1)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'judge_' + j + '.json';
  a.click();
  document.getElementById('status').textContent = '已导出 judge_' + j + '.json';
}}

build();
save();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
