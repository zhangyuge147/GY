# -*- coding: utf-8 -*-
"""
光影重生 · 盲评评分回收汇总（任务 3.11）
========================================
读取 3 名评分员导出的 judge_<编号>.json（问卷作答后浏览器下载），汇总：

  1. 修复后 vs 修复前平均质量分（1-5）：每样本取"修复后"图的分数与"修复前"
     图的分数，做配对样本（3 人 × 12 组 = 36 对）均值对比 + 配对 t 检验。
  2. 偏好胜出率：A/B/差不多 三选一中指向"修复后"的比例（差不多按 0.5 加权）。
  3. 按类别（building/rural/scene/portrait）分组汇总。

用法（在 guangying/ 目录、guangying 环境下）：
  python evaluation/blind_judge/aggregate_judges.py
  python evaluation/blind_judge/aggregate_judges.py --dir data/blind_responses
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

ROOT = Path(__file__).resolve().parent.parent       # evaluation/
DEFAULT_DIR = ROOT / "blind_judge"
CATS = ("building", "rural", "scene", "portrait")


def fmt(v) -> str:
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description="光影重生·盲评评分汇总（任务 3.11）")
    ap.add_argument("--dir", type=str, default=str(DEFAULT_DIR),
                    help="judge_*.json 所在目录（默认 blind_judge/）")
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "data" / "blind_judge_summary.csv"))
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("judge_*.json"))
    if not files:
        raise SystemExit(f"[!] {args.dir} 下没有 judge_*.json，请先回收评分员问卷"
                         "（问卷作答后浏览器下载评分 JSON）")
    print(f"回收评分：{len(files)} 份 {[f.name for f in files]}")

    # 逐份解析：meta 提供 A/B 是否修复后；answers 提供分数与偏好
    pairs, wins = [], []        # pairs: (rep_score, base_score)；wins: 偏好指向修复后
    per_sample = {}             # id -> {rep[], base[], pref[]}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        meta = {m["id"]: m for m in d.get("meta", [])}
        for gid, a in (d.get("answers") or {}).items():
            m = meta.get(gid)
            if not m or not a.get("scoreA") or not a.get("scoreB"):
                continue
            rep_score = a["scoreA"] if m["A_is_repair"] else a["scoreB"]
            base_score = a["scoreA"] if m["B_is_repair"] else a["scoreB"]
            pairs.append((rep_score, base_score))
            pref = a.get("pref")
            if pref == "tie":
                win = 0.5
            elif (pref == "A") == m["A_is_repair"]:
                win = 1.0
            else:
                win = 0.0
            wins.append(win)
            s = per_sample.setdefault(gid, {"id": gid, "cat": m["cat"],
                                           "rep": [], "base": [], "win": []})
            s["rep"].append(rep_score)
            s["base"].append(base_score)
            s["win"].append(win)

    if not pairs:
        raise SystemExit("[!] 未解析到任何有效评分（answers 为空或字段缺失）")

    # 全局配对对比
    n = len(pairs)
    rep_mean = statistics.mean(p[0] for p in pairs)
    base_mean = statistics.mean(p[1] for p in pairs)
    diff = [p[0] - p[1] for p in pairs]
    p_value = None
    if HAS_SCIPY and n > 1:
        p_value = float(sp_stats.ttest_rel([p[0] for p in pairs],
                                           [p[1] for p in pairs]).pvalue)
    win_rate = statistics.mean(wins)

    print("\n=== 全局（3 人 × %d 组 = %d 个配对评分）===" % (len(per_sample), n))
    print(f"修复后平均分 {fmt(rep_mean)}  vs  修复前平均分 {fmt(base_mean)}"
          f"  Δ {fmt(rep_mean - base_mean)}")
    if p_value is not None:
        sig = "显著" if p_value < 0.05 else "不显著"
        print(f"配对 t 检验 p = {p_value:.4f}（{sig}）")
    print(f"偏好胜出率（指向修复后）{fmt(win_rate * 100)}%"
          f"（n={n}，'差不多'按 0.5 加权）")
    if not HAS_SCIPY:
        print("(scipy 未安装，跳过显著性检验)")

    # 分样本 + 类别汇总
    print("\n=== 分样本（3 人均值）===")
    print(f"{'样本':<16} {'类别':<10} {'修复后':>7} {'修复前':>7} {'Δ':>7}"
          f" {'偏好胜出':>8}")
    rows = []
    cat_rows = {}
    for gid in sorted(per_sample):
        s = per_sample[gid]
        r, b, w = statistics.mean(s["rep"]), statistics.mean(s["base"]), statistics.mean(s["win"])
        rows.append([gid, s["cat"], fmt(r), fmt(b), fmt(r - b), f"{w * 100:.1f}%"])
        cat_rows.setdefault(s["cat"], []).append((r, b, w))
        print(f"{gid:<16} {s['cat']:<10} {fmt(r):>7} {fmt(b):>7} {fmt(r - b):>7}"
              f" {w * 100:>7.1f}%")

    print("\n=== 按类别汇总（均值）===")
    for cat in CATS + ("other",):
        data = cat_rows.get(cat, [])
        if not data:
            continue
        r = statistics.mean(x[0] for x in data)
        b = statistics.mean(x[1] for x in data)
        w = statistics.mean(x[2] for x in data)
        print(f"{cat:<10} n={len(data):>2} | 修复后 {fmt(r)}  修复前 {fmt(b)}"
              f"  Δ {fmt(r - b)}  胜出率 {w * 100:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "类别", "修复后均分", "修复前均分", "Δ分", "偏好胜出率%"])
        w.writerows(rows)
        w.writerow(["", "总体", fmt(rep_mean), fmt(base_mean),
                    fmt(rep_mean - base_mean), f"{win_rate * 100:.1f}"])
    print(f"\n结果已写入：{out}")


if __name__ == "__main__":
    main()
