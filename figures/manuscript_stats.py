#!/usr/bin/env python3
"""Every statistic quoted in the manuscript, recomputed from results/, printed by section.

Definitions match figures/make_figures.py (same model order, same Exp-3 measures),
so a number in the text and the figure it points to always come from one source.

Usage:  python3 figures/manuscript_stats.py
"""
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import parse_decision  # noqa: E402

csv.field_size_limit(sys.maxsize)
SEV = {"accept": 0, "minor_revision": 1, "major_revision": 2, "reject": 3}
MODELS = ["openai/gpt-5.6-sol", "z-ai/glm-5.3-flash", "google/gemini-3.7-flash", "deepseek/deepseek-v4-flash-0731"]
NAME = dict(zip(MODELS, ["GPT-5.6 Sol", "GLM-5.3 Flash", "Gemini 3.7 Flash", "DeepSeek v4 Flash"]))
CORPUS = set(json.loads((ROOT / "results" / "final_corpus_dois.json").read_text()))
JOURNAL = json.loads((ROOT / "papers" / "_journal_manifest.json").read_text())


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)"


def p(x):
    return "p < 0.001" if x < 0.001 else f"p = {x:.3g}"


def wide(path, dedupe_last=False):
    d = pd.read_csv(ROOT / "results" / path)
    d["s"] = d.decision.map(SEV)
    d = d.dropna(subset=["s"])
    if dedupe_last:  # retried rows: keep the successful (last) one per pair
        d = d.drop_duplicates(["doi", "model"], keep="last")
    assert set(d.doi) == CORPUS, path
    return d, d.pivot(index="doi", columns="model", values="s")[MODELS]


def kendall_w(w):
    """Kendall's W with tie correction; raters = models (columns), objects = manuscripts (rows)."""
    m, n = w.shape[1], w.shape[0]
    r = w.rank(axis=0)
    s = ((r.sum(axis=1) - r.sum(axis=1).mean()) ** 2).sum()
    t = sum(((c := w[col].value_counts().values) ** 3 - c).sum() for col in w)
    return 12 * s / (m ** 2 * (n ** 3 - n) - m * t)


def section(title):
    print(f"\n=== {title}")


# ------------------------------------------------------------------ corpus
section("§2.1 corpus")
print("journals:", len(set(JOURNAL[d] for d in CORPUS)), "| per journal:", set(Counter(JOURNAL[d] for d in CORPUS).values()))

# ------------------------------------------------------------------ Exp 1
section("§3.1 / abstract: decisions by model")
e1, w1 = wide("exp1_decisions.csv")
mr = {m: int((w1[m] >= 2).sum()) for m in MODELS}
for m in MODELS:
    print(f"  {NAME[m]:18} major/reject {pct(mr[m], 200)}  dist {w1[m].value_counts().sort_index().to_dict()}  mean {w1[m].mean():.2f}")
print("  pooled major/reject", pct(sum(mr.values()), 800), "| distribution", {k: f"{(e1.s == v).mean() * 100:.1f}%" for k, v in SEV.items()})
H, kp = stats.kruskal(*[w1[m] for m in MODELS])
print(f"  Kruskal-Wallis H = {H:.1f}, {p(kp)}, eps2 = {H / ((800 ** 2 - 1) / 801):.2f}")
fr = stats.friedmanchisquare(*[w1[m] for m in MODELS])
print(f"  Friedman chi2 = {fr.statistic:.1f}, {p(fr.pvalue)}")
rhos = {(a, b): stats.spearmanr(w1[a], w1[b]) for a, b in combinations(MODELS, 2)}
print(f"  Kendall's W (models ranking manuscripts, tie-corrected) = {kendall_w(w1):.2f}")
print("  pairwise Spearman rho:", ", ".join(f"{NAME[a].split()[0]}-{NAME[b].split()[0]} {r.statistic:.2f} ({p(r.pvalue)})" for (a, b), r in rhos.items()))
print(f"  rho range {min(r.statistic for r in rhos.values()):.2f}-{max(r.statistic for r in rhos.values()):.2f}")

section("§3.1 repeat review (intra-model) vs cross-model")
_, w2 = wide("exp1_intra_rater_retest.csv", dedupe_last=True)
w2 = w2.loc[w1.index]
for m in MODELS:
    others = {NAME[o]: (round((w1[m] == w1[o]).mean() * 100, 1), round(cohen_kappa_score(w1[m], w1[o]), 2)) for o in MODELS if o != m}
    print(f"  {NAME[m]:18} self: agree {(w1[m] == w2[m]).mean() * 100:.1f}%, kappa {cohen_kappa_score(w1[m], w2[m]):.2f}, "
          f"wkappa {cohen_kappa_score(w1[m], w2[m], weights='quadratic'):.2f} | vs others (agree%, kappa): {others}")

section("§3.1 unanimity and journals")
un = w1[w1.nunique(axis=1) == 1]
print(f"  unanimous {pct(len(un), 200)}, decisions {Counter(un.iloc[:, 0]).most_common()}, journals spanned {len({JOURNAL[d] for d in un.index})}")
print(f"  four distinct decisions: {(w1.nunique(axis=1) == 4).sum()}")
e1["journal"] = e1.doi.map(JOURNAL)
Hj, pj = stats.kruskal(*[g.s for _, g in e1.groupby("journal")])
print(f"  journals, review-level: Kruskal-Wallis H = {Hj:.1f}, {p(pj)} (40 reviews each)")
mj = e1.groupby(["journal", "doi"]).s.mean().reset_index()
Hm, pm = stats.kruskal(*[g.s for _, g in mj.groupby("journal")])
print(f"  journals, manuscript-level (mean of 4 models): H = {Hm:.1f}, {p(pm)}")
jm = e1.groupby("journal").s.mean().sort_values()
print("  journal means:", {k: round(v, 2) for k, v in jm.items()})
top = e1[e1.journal.isin(["Nature", "Science", "Cell"])]
sr = e1[e1.journal == "Scientific Reports"]
print(f"  Nature/Science/Cell mean {top.s.mean():.2f} (n = {len(top)}) vs Sci Rep {sr.s.mean():.2f} (n = {len(sr)}), "
      f"Mann-Whitney {p(stats.mannwhitneyu(top.s, sr.s).pvalue)}")

# ------------------------------------------------------------------ revision framing
section("§3.2 revision framing")
_, w3 = wide("exp1_secondround.csv")
w3 = w3.loc[w1.index]
d1, d3 = w1.values.ravel(), w3.values.ravel()
diff = d3 - d1
print(f"  pooled mean {d1.mean():.2f} -> {d3.mean():.2f} (n = {diff.size}), Wilcoxon {p(stats.wilcoxon(d3, d1).pvalue)}")
print(f"  milder {pct(int((diff < 0).sum()), diff.size)}, unchanged {pct(int((diff == 0).sum()), diff.size)}, harsher {pct(int((diff > 0).sum()), diff.size)}")
for m in MODELS:
    print(f"  {NAME[m]:18} shift {(w3[m] - w1[m]).mean():+.2f}, Wilcoxon {p(stats.wilcoxon(w3[m], w1[m]).pvalue)}, "
          f"accept {(w1[m] == 0).mean() * 100:.1f}% -> {(w3[m] == 0).mean() * 100:.1f}%, "
          f"modal {w1[m].mode()[0]:.0f} -> {w3[m].mode()[0]:.0f}")
for k, v in SEV.items():
    print(f"  {k:15} {(d1 == v).mean() * 100:.1f}% -> {(d3 == v).mean() * 100:.1f}%")

# ------------------------------------------------------------------ Exp 2
section("§2.3 / §3.3 detection")
e2 = pd.DataFrame([json.loads(l) for l in open(ROOT / "results" / "exp2_injections.jsonl") if l.strip()])
assert set(e2.doi) == CORPUS and len(e2) == 800
e2["v"] = e2.detected.map(lambda x: "yes" if x is True else ("partial" if x == "partial" else "no"))
print("  verdicts", e2.v.value_counts().to_dict(), f"| partial share {100 * (e2.v == 'partial').mean():.1f}%")
ct = pd.crosstab(e2.reviewer_model, e2.v == "yes").loc[MODELS]
chi2, cp_, _, _ = stats.chi2_contingency(ct)
for m in MODELS:
    x = e2[e2.reviewer_model == m]
    print(f"  {NAME[m]:18} caught {pct(int((x.v == 'yes').sum()), len(x))}, with partial {100 * (x.v != 'no').mean():.1f}%, "
          f"stat {100 * (x[x.category == 'statistical'].v == 'yes').mean():.1f}% / meth {100 * (x[x.category == 'methodological'].v == 'yes').mean():.1f}%")
print(f"  chi2 = {chi2:.1f}, {p(cp_)}")
for c in ("statistical", "methodological"):
    x = e2[e2.category == c]
    print(f"  {c}: {pct(int((x.v == 'yes').sum()), len(x))}")
cc = stats.chi2_contingency(pd.crosstab(e2.category, e2.v == "yes"))
print(f"  category chi2 = {cc[0]:.2f}, {p(cc[1])}")
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    g = e2.assign(y=(e2.v == "yes").astype(int), model=e2.reviewer_model.map(NAME))
    gee = smf.gee("y ~ C(model, Treatment('DeepSeek v4 Flash'))", "doi", g, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    print("  GEE (ref DeepSeek) p:", {k.split("T.")[-1].rstrip("]"): round(v, 4) for k, v in gee.pvalues.items() if "T." in k})
except Exception as exc:  # noqa: BLE001
    print("  GEE failed:", exc)

section("§3.3 severity shift after injection (paired with Exp 1)")
e2["s2"] = [SEV.get(parse_decision.parse_decision(t)[0]) for t in e2.raw_review]
long1 = e1[["doi", "model", "s"]].rename(columns={"model": "reviewer_model"})
pr = e2.merge(long1, on=["doi", "reviewer_model"])
pr = pr.dropna(subset=["s2"])
pr["d"] = pr.s2 - pr.s
print(f"  pooled shift {pr.d.mean():+.3f} (n = {len(pr)}), Wilcoxon {p(stats.wilcoxon(pr.s2, pr.s).pvalue)}")
for m in MODELS:
    x = pr[pr.reviewer_model == m]
    print(f"  {NAME[m]:18} {x.d.mean():+.3f} (n = {len(x)}), Wilcoxon {p(stats.wilcoxon(x.s2, x.s).pvalue)}, "
          f"major/reject {(x.s >= 2).sum()} -> {(x.s2 >= 2).sum()}")
for v in ("yes", "no", "partial"):
    x = pr[pr.v == v]
    print(f"  judged {v:7} n = {len(x)}, shift {x.d.mean():+.3f}, Wilcoxon {p(stats.wilcoxon(x.s2, x.s).pvalue)}, unchanged {100 * (x.d == 0).mean():.1f}%")

# ------------------------------------------------------------------ Exp 3
section("§3.4 judging")
J = {}
for key, path in (("grok", "exp5_review_quality.jsonl"), ("minimax", "exp5_review_quality_minimax.jsonl")):
    recs = {r["doi"]: r for r in (json.loads(l) for l in open(ROOT / "results" / path) if l.strip())}
    assert set(recs) == CORPUS, path
    J[key] = recs
    wins = Counter(r["best_review_model"] for r in recs.values())
    gof = stats.chisquare([wins.get(m, 0) for m in MODELS])
    print(f"  [{key}] best: " + ", ".join(f"{NAME[m]} {pct(wins.get(m, 0), 200)}" for m in MODELS) + f" | GOF chi2 = {gof.statistic:.1f}, {p(gof.pvalue)}")
    rows = []
    for r in recs.values():
        l2m = r["letter_to_model"]
        for c in r["concern_clusters"]:
            ms = {l2m[l] for l in c.get("raised_by", []) if l in l2m}
            if ms and c.get("category") in ("relevant", "nitpicking"):
                rows.append((len(ms), c["category"] == "nitpicking", ms))
    k = pd.DataFrame(rows, columns=["n", "nit", "ms"])
    tab = pd.crosstab(k.n, k.nit)
    chi = stats.chi2_contingency(tab)
    rho = stats.spearmanr(k.n, k.nit)
    print(f"  [{key}] clusters {len(k)}, nitpicking by #reviewers {k.groupby('n').nit.mean().mul(100).round(1).to_dict()}, "
          f"chi2 = {chi[0]:.1f}, {p(chi[1])}, Spearman rho = {rho.statistic:.2f} ({p(rho.pvalue)})")
    for m in MODELS:
        letters = lambda r: {l for l, mm in r["letter_to_model"].items() if mm == m}  # noqa: E731
        rel = [sum(1 for c in r["concern_clusters"] if c.get("category") == "relevant" and set(c.get("raised_by", [])) & letters(r)) for r in recs.values()]
        con = [r["constructiveness"][l] for r in recs.values() for l in letters(r) if r["constructiveness"].get(l) is not None]
        mine = k[k.ms.map(lambda s: m in s)]
        print(f"     {NAME[m]:18} relevant/manuscript {np.mean(rel):.2f}, constructiveness {np.mean(con):.2f}, nitpicking {100 * mine.nit.mean():.1f}%")
common = sorted(CORPUS)
bg = [J["grok"][d]["best_review_model"] for d in common]
bm = [J["minimax"][d]["best_review_model"] for d in common]
print(f"  judges agree on best: {100 * np.mean([a == b for a, b in zip(bg, bm)]):.1f}%, kappa = {cohen_kappa_score(bg, bm):.2f}")
cg, cm = [], []
for d in common:
    for key, out in (("grok", cg), ("minimax", cm)):
        r = J[key][d]
        by_model = {mm: r["constructiveness"][l] for l, mm in r["letter_to_model"].items()}
        out.extend(by_model[m] for m in MODELS)
print(f"  constructiveness, judge vs judge: Spearman rho = {stats.spearmanr(cg, cm).statistic:.2f} over {len(cg)} review-level scores")

section("Figure 7: relevant-concern overlap (grok)")
own = Counter()
pair = Counter()
for r in J["grok"].values():
    l2m = r["letter_to_model"]
    for c in r["concern_clusters"]:
        if c["category"] != "relevant":
            continue
        ms = sorted({l2m[l] for l in c["raised_by"] if l in l2m})
        own.update(ms)
        pair.update(frozenset(x) for x in combinations(ms, 2))
print("  own relevant concerns:", {NAME[m]: own[m] for m in MODELS})
for a in MODELS:
    print(f"  {NAME[a]:18} covered by: " + ", ".join(f"{NAME[b]} {100 * pair[frozenset((a, b))] / own[a]:.0f}%" for b in MODELS if b != a))
