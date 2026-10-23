#!/usr/bin/env python3
"""
Static Nature-style PNG figures for the LLM peer review pilot.
Fig 1: decision distribution by model (Exp 1), with Kruskal-Wallis omnibus p
Fig 2: every paper x model decision, full heatmap (Exp 1)
Fig 3: journal-swap effect, paired true-vs-swapped-label severity, both
       directions, with per-row Wilcoxon signed-rank p (Exp 4)
Fig 4: flaw detection rate by model (Exp 2), caught vs missed, with
       chi-square omnibus p
"""

import csv
import json
from decimal import Decimal, ROUND_HALF_UP
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, kruskal, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.dpi"] = 150

GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"
SEV_COLORS = [GOOD, WARNING, SERIOUS, CRITICAL]
SEV_KEYS = ["accept", "minor_revision", "major_revision", "reject"]
SEV = {k: i for i, k in enumerate(SEV_KEYS)}
INK = "#16140d"
MUTED = "#6b6a62"
RULE = "#dbd7c9"

MODEL_LABEL = {
    "google/gemini-3.7-flash": "Gemini 3.7 Flash",
    "z-ai/glm-5.3-flash": "GLM-5.3 Flash",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek v4 Flash",
}
MODEL_SHORT = {
    "google/gemini-3.7-flash": "Gemini 3.7",
    "z-ai/glm-5.3-flash": "GLM-5.3",
    "openai/gpt-5.6-sol": "GPT-5.6",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek v4",
}
MODELS = ["openai/gpt-5.6-sol", "z-ai/glm-5.3-flash", "google/gemini-3.7-flash", "deepseek/deepseek-v4-flash-0731"]

PREFIX_JOURNAL = {
    "10.1038/s41586": "Nature",
    "10.1038/s41591": "Nature Medicine",
    "10.1038/s41467": "Nature Communications",
    "10.1038/s41588": "Nature Genetics",
    "10.1038/s41598": "Scientific Reports",
    "10.1016/j.xcrm": "Cell Reports Medicine",
    "10.1371/journal.pmed": "PLoS Medicine",
}
JOURNAL_ORDER = [
    "Nature", "Nature Medicine", "Nature Communications", "Nature Genetics",
    "Nature Materials", "Scientific Reports",
    "Science", "Cell", "Cell Reports Medicine", "PLoS Biology", "PLoS Medicine", "eLife",
    "PNAS",
    "Circulation", "European Heart Journal",
    "Journal of Experimental Medicine", "JAMA Internal Medicine",
    "JAMA Oncology", "JAMA Psychiatry",
    "Lancet Global Health",
]
JOURNAL_SHORT = {
    "Nature": "Nature", "Nature Medicine": "Nat. Med.", "Nature Communications": "Nat. Comms.",
    "Nature Genetics": "Nat. Genet.",
    "Nature Materials": "Nat. Mater.", "Scientific Reports": "Sci. Rep.", "Science": "Science",
    "Cell": "Cell", "Cell Reports Medicine": "Cell Rep. Med.", "PLoS Biology": "PLoS Biol.",
    "PLoS Medicine": "PLoS Med.", "eLife": "eLife",
    "PNAS": "PNAS",
    "Circulation": "Circulation", "European Heart Journal": "Eur. Heart J.",
    "Journal of Experimental Medicine": "J Exp Med", "JAMA Internal Medicine": "JAMA IM",
    "JAMA Oncology": "JAMA Onc.", "JAMA Psychiatry": "JAMA Psych.",
    "Lancet Global Health": "Lancet GH",
}

manifest = json.loads((ROOT / "papers" / "_journal_manifest.json").read_text())



def _panel(ax, letter, x=0.0, y=1.06):
    """Nature-style panel label: bold lowercase letter, outside the axes, no title."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            color=INK, ha="left", va="bottom")


def journal_for(doi):
    if doi in manifest:
        return manifest[doi]
    for prefix, j in PREFIX_JOURNAL.items():
        if doi.startswith(prefix):
            return j
    return "?"


def p_str(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


# ---------------------------------------------------------------- Fig 1
def fig1_status_by_model():
    with open(ROOT / "results" / "exp1_decisions.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] != "anthropic/claude-opus-5"]

    # Ordered by descending share of major revision + reject, so the panel reads
    # from harshest to most lenient (GPT > DeepSeek > GLM > Gemini).
    models = MODELS
    data = []
    sev_by_model = []
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        counts = [sum(1 for r in mrows if r["decision"] == k) for k in SEV_KEYS]
        n = len(mrows)
        data.append([c / n * 100 for c in counts])
        sev_by_model.append([SEV[r["decision"]] for r in mrows])

    kw_p = kruskal(*sev_by_model).pvalue

    fig, ax = plt.subplots(figsize=(5.2, 2.65))
    y = np.arange(len(models))[::-1]
    left = np.zeros(len(models))
    for i, key in enumerate(SEV_KEYS):
        vals = [d[i] for d in data]
        ax.barh(y, vals, left=left, height=0.58, color=SEV_COLORS[i], edgecolor="white", linewidth=1.2)
        for yi, v, l in zip(y, vals, left):
            if v >= 6:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if i in (0, 3) else INK)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in models], fontsize=7.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, colors=INK, labelsize=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEV_COLORS]
    labels = ["Accept", "Minor revision", "Major revision", "Reject"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=4,
              frameon=False, fontsize=6.5, handlelength=1.0, handleheight=1.0, columnspacing=1.0)

    print("fig1 kruskal p:", kw_p)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_status_by_model.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_status_by_model.png")


# ---------------------------------------------------------------- Fig 2
def fig2_status_all_papers(n_blocks=5):
    with open(ROOT / "results" / "exp1_decisions.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] != "anthropic/claude-opus-5"]

    models = MODELS
    by_doi = {}
    for r in rows:
        by_doi.setdefault(r["doi"], {})[r["model"]] = SEV[r["decision"]]

    papers = []
    for doi, cells in by_doi.items():
        j = journal_for(doi)
        mean_sev = np.mean(list(cells.values()))
        papers.append((j, mean_sev, doi, cells))
    papers.sort(key=lambda p: (JOURNAL_ORDER.index(p[0]) if p[0] in JOURNAL_ORDER else 99, -p[1]))

    cmap = matplotlib.colors.ListedColormap(SEV_COLORS)

    # split journals into n_blocks column-groups, each drawn as its own mini-heatmap panel
    journals_per_block = int(np.ceil(len(JOURNAL_ORDER) / n_blocks))
    blocks = [JOURNAL_ORDER[i:i + journals_per_block] for i in range(0, len(JOURNAL_ORDER), journals_per_block)]

    cell = 1.0       # cell pitch
    pad = 0.16       # gap between cells (fraction of pitch removed from each side)
    sq = cell - 2 * pad
    GROUP_GAP = 0.45  # extra vertical space inserted between journal groups, in row-pitch units
    BAND_COLORS = ["#f2f0e9", "#ffffff"]  # alternating journal-group background bands

    def group_boundaries(journals_in_order):
        n = len(journals_in_order)
        starts = [0] + [i for i in range(1, n) if journals_in_order[i] != journals_in_order[i - 1]]
        starts.append(n)
        return starts

    def row_positions(journals_in_order):
        """Cumulative row position for each paper, inserting GROUP_GAP at each
        journal boundary, so groups are visibly separated. Returns (positions, extent)."""
        positions = []
        extra = 0.0
        for i, j in enumerate(journals_in_order):
            if i > 0 and j != journals_in_order[i - 1]:
                extra += GROUP_GAP
            positions.append(i + extra)
        extent = (positions[-1] + 1) if positions else 0
        return positions, extent

    block_papers = []
    block_positions = []
    max_extent = 0.0
    for block_journals in blocks:
        bp = [p for p in papers if p[0] in block_journals]
        block_papers.append(bp)
        journals_in_order = [p[0] for p in bp]
        positions, extent = row_positions(journals_in_order)
        block_positions.append(positions)
        max_extent = max(max_extent, extent)

    LEFT_MARGIN = -5.4   # room for full (wrapped, multi-line) journal names
    HEADER_ROOM = 2.6    # vertical room above the top row for rotated model-name headers
    LABEL_WRAP_WIDTH = 16  # characters per line before wrapping a journal name

    # scale figure width so the wider label margin doesn't shrink the (equal-aspect) cells --
    # 2.9in per block was calibrated against the old xlim range of 7.6 units (-3.6 to 4)
    xlim_range = (len(MODELS) - LEFT_MARGIN)
    fig_w = 2.9 * n_blocks * (xlim_range / 7.6)
    fig_h = max(3.0, max_extent * 0.16 + 1.3)
    fig, axes = plt.subplots(1, n_blocks, figsize=(fig_w, fig_h))
    if n_blocks == 1:
        axes = [axes]

    for ax, bp, positions in zip(axes, block_papers, block_positions):
        n = len(bp)
        journals_in_order = [p[0] for p in bp]
        # top-align every block against the same reference (max_extent), so a
        # shorter block (fewer journals/papers) simply leaves blank space at
        # the bottom instead of being vertically centered against the tallest one
        def row_y(i):
            return max_extent - 1 - positions[i]

        group_starts = group_boundaries(journals_in_order)

        # alternating background band per journal group, drawn behind the cells.
        # row i occupies the unit slot [row_y(i), row_y(i)+1]; a small margin
        # (less than half of GROUP_GAP) keeps adjacent bands from touching.
        band_margin = 0.12
        for gi in range(len(group_starts) - 1):
            start, end = group_starts[gi], group_starts[gi + 1]
            y_top = row_y(start) + 1 + band_margin
            y_bot = row_y(end - 1) - band_margin
            ax.add_patch(plt.Rectangle((LEFT_MARGIN, y_bot), len(models) - LEFT_MARGIN, y_top - y_bot,
                                        facecolor=BAND_COLORS[gi % 2], edgecolor="none", zorder=0))

        for i, (j, _, doi, cells) in enumerate(bp):
            ry = row_y(i)
            for mi, m in enumerate(models):
                v = cells.get(m)
                color = SEV_COLORS[v] if v is not None else "#f1f0ea"
                ax.add_patch(plt.Rectangle((mi + pad, ry + pad), sq, sq,
                                            facecolor=color, edgecolor="none", zorder=2))

        # model names on top, aligned the same way in every block, with clear
        # air between the header text and the top row of cells
        header_y = max_extent + 0.5
        for mi, m in enumerate(models):
            ax.text(mi + 0.5, header_y, MODEL_SHORT[m], ha="center", va="bottom",
                    fontsize=7.2, color=INK, rotation=90, zorder=3)

        # journal group labels on the left -- full name, wrapped onto multiple
        # lines rather than abbreviated, so nothing gets truncated or guessed at
        for gi in range(len(group_starts) - 1):
            start, end = group_starts[gi], group_starts[gi + 1]
            mid_row = (row_y(start) + 1 + row_y(end - 1)) / 2
            full_name = journals_in_order[start]
            wrapped = textwrap.fill(full_name, width=LABEL_WRAP_WIDTH, break_long_words=False)
            ax.text(-0.45, mid_row, wrapped, ha="right", va="center", fontsize=7.0, color=INK,
                    linespacing=1.25, multialignment="right", zorder=3)

        ax.set_xlim(LEFT_MARGIN, len(models))
        ax.set_ylim(-0.5, max_extent + HEADER_ROOM)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal")

    # shared discrete legend (decisions are categorical, not a continuous scale)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEV_COLORS]
    labels = ["Accept", "Minor revision", "Major revision", "Reject"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4,
               frameon=False, fontsize=6.5, handlelength=1.0, handleheight=1.0, columnspacing=1.0)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_status_all_papers.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig2_status_all_papers.png ({len(papers)} papers, {n_blocks} blocks)")


# ---------------------------------------------------------------- Fig 3
def fig3_journal_swap():
    with open(ROOT / "results" / "exp1_decisions.csv") as f:
        rows1 = [r for r in csv.DictReader(f) if r["model"] != "anthropic/claude-opus-5"]
    with open(ROOT / "results" / "exp4_journal_swap.csv") as f:
        rows4 = list(csv.DictReader(f))

    true_label = {(r["doi"], r["model"]): SEV[r["decision"]] for r in rows1
                  if "s41586" in r["doi"] or "s41598" in r["doi"]}
    swap_label = {(r["doi"], r["model"]): SEV[r["decision"]] for r in rows4}

    models = MODELS
    # (row label, filter, true_col_is_nature)
    directions = [
        ("Sci. Reports → Nature", "s41598", False),  # real Sci Rep, true=SciRep, swap=Nature
        ("Nature → Sci. Reports", "s41586", True),   # real Nature, true=Nature, swap=SciRep
    ]

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    row_labels = []
    y = 0
    yticks = []
    for m in models:
        for label, prefix, true_is_nature in directions:
            true_vals, swap_vals = [], []
            for (doi, model), t in true_label.items():
                if model != m or prefix not in doi:
                    continue
                key = (doi, model)
                if key not in swap_label:
                    continue
                true_vals.append(t)
                swap_vals.append(swap_label[key])
            true_vals = np.array(true_vals)
            swap_vals = np.array(swap_vals)
            diffs = swap_vals - true_vals
            if np.any(diffs != 0):
                _, p = wilcoxon(diffs)
            else:
                p = 1.0

            start = true_vals.mean()
            end = swap_vals.mean()
            arrow_color = CRITICAL if not true_is_nature else GOOD
            ax.annotate(
                "", xy=(end, y), xytext=(start, y),
                arrowprops=dict(arrowstyle="-|>", color=arrow_color, linewidth=1.8,
                                 mutation_scale=12, shrinkA=0, shrinkB=0),
            )
            ax.scatter([start], [y], s=26, color="white", edgecolor=INK, linewidth=1.0, zorder=3)
            sig = "*" if p < 0.05 else ""
            ax.text(3.15, y, f"{p_str(p)}{sig}", fontsize=6.2, color=MUTED, va="center", ha="left")
            row_labels.append(f"{MODEL_LABEL[m]}\n{label}")
            yticks.append(y)
            y -= 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(row_labels, fontsize=6.3, color=INK)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(0, 4.35)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Accept", "Minor", "Major", "Reject"], fontsize=6.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(left=False, colors=INK)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5)
    for x in [0, 1, 2, 3]:
        ax.axvline(x, color=RULE, linewidth=0.5, zorder=0)

    handles = [
        plt.Line2D([0], [0], color=CRITICAL, linewidth=1.8, marker=">", markersize=5, label="Relabeled Nature"),
        plt.Line2D([0], [0], color=GOOD, linewidth=1.8, marker=">", markersize=5, label="Relabeled Sci. Reports"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, 1.14), ncol=2,
              frameon=False, fontsize=6.3, handletextpad=0.4, columnspacing=1.1)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_journal_swap.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_journal_swap.png")


# ---------------------------------------------------------------- Fig 4
def fig4_exp2_detection(by_category=True):
    records = [json.loads(l) for l in open(ROOT / "results" / "exp2_injections.jsonl") if l.strip()]
    models = MODELS
    CATEGORY_LABEL = {"statistical": "Statistical", "methodological": "Methodological"}
    categories = ["statistical", "methodological"]

    table = []
    for m in models:
        mrows = [r for r in records if r["reviewer_model"] == m]
        caught = sum(1 for r in mrows if r["detected"] is True)
        table.append([caught, len(mrows) - caught])
    chi2, chi_p, _, _ = chi2_contingency(table)

    if not by_category:
        fig, ax = plt.subplots(figsize=(4.4, 2.3))
        models_s, table_s = models, table
        y = np.arange(len(models_s))[::-1]
        for yi, m, (caught, missed) in zip(y, models_s, table_s):
            n = caught + missed
            pct = caught / n * 100
            ax.barh([yi], [pct], height=0.58, color=GOOD, edgecolor="white", linewidth=1.2)
            ax.barh([yi], [100 - pct], left=[pct], height=0.58, color=CRITICAL, edgecolor="white", linewidth=1.2)
            ax.text(pct / 2, yi, f"{pct:.0f}%", ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
            ax.text(pct + (100 - pct) / 2, yi, f"{100 - pct:.0f}%", ha="center", va="center", fontsize=6.8,
                    color="white", fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_LABEL[m] for m in models_s], fontsize=7.5, color=INK)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7)
        for s in ("left", "bottom", "top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(left=False, colors=INK, labelsize=7)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=RULE, linewidth=0.5)

        handles = [plt.Rectangle((0, 0), 1, 1, color=GOOD), plt.Rectangle((0, 0), 1, 1, color=CRITICAL)]
        ax.legend(handles, ["Caught", "Missed"], loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2,
                  frameon=False, fontsize=6.5, handlelength=1.0, handleheight=1.0, columnspacing=1.0)


        fig.tight_layout()
        fig.savefig(OUT / "fig4_exp2_detection.png", bbox_inches="tight")
        plt.close(fig)
        print("wrote fig4_exp2_detection.png")
        return

    # by_category=True: two rows per model (statistical, methodological), grouped with a gap between models
    rows = []  # (model, category, caught, n)
    for m in models:
        for cat in categories:
            mrows = [r for r in records if r["reviewer_model"] == m and r["category"] == cat]
            caught = sum(1 for r in mrows if r["detected"] is True)
            rows.append((m, cat, caught, len(mrows)))

    n_rows = len(rows)
    group_gap = 0.55
    y_positions = []
    y = 0.0
    for i, (m, cat, caught, n) in enumerate(rows):
        y_positions.append(y)
        y -= 1.0
        if cat == categories[-1] and i < n_rows - 1:
            y -= group_gap
    y_positions = np.array(y_positions)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for yi, (m, cat, caught, n) in zip(y_positions, rows):
        pct = caught / n * 100
        ax.barh([yi], [pct], height=0.68, color=GOOD, edgecolor="white", linewidth=1.0)
        ax.barh([yi], [100 - pct], left=[pct], height=0.68, color=CRITICAL, edgecolor="white", linewidth=1.0)
        ax.text(pct / 2, yi, f"{pct:.0f}%", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
        ax.text(pct + (100 - pct) / 2, yi, f"{100 - pct:.0f}%", ha="center", va="center", fontsize=6.6,
                color="white", fontweight="bold")

    # category tick labels on the axis itself
    ax.set_yticks(y_positions)
    ax.set_yticklabels([CATEGORY_LABEL[cat] for _, cat, _, _ in rows], fontsize=6.8, color=MUTED)

    # model group labels to the left, vertically centered on each pair of rows
    for i in range(0, n_rows, len(categories)):
        group_y = np.mean(y_positions[i:i + len(categories)])
        m = rows[i][0]
        ax.text(-14, group_y, MODEL_LABEL[m], ha="right", va="center", fontsize=8.2, color=INK, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(y_positions.min() - 0.7, y_positions.max() + 0.7)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False, colors=INK, labelsize=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GOOD), plt.Rectangle((0, 0), 1, 1, color=CRITICAL)]
    ax.legend(handles, ["Caught", "Missed"], loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2,
              frameon=False, fontsize=7, handlelength=1.0, handleheight=1.0, columnspacing=1.0)

    print("fig4 chi2 across models:", chi2, chi_p)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_exp2_detection.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig4_exp2_detection.png (by category)")


MODEL_COLOR = {
    # Categorical, CVD-safe (all-pairs, validated), deliberately distinct from the
    # GOOD/WARNING/SERIOUS/CRITICAL status palette used for decision severity and
    # caught/missed elsewhere, so a color never carries two different meanings across figures.
    "z-ai/glm-5.3-flash": "#2a78d6",          # blue
    "openai/gpt-5.6-sol": "#eb6834",          # orange
    "google/gemini-3.7-flash": "#1baf7a",     # aqua
    "deepseek/deepseek-v4-flash-0731": "#4a3aa7",  # violet
}


# ---------------------------------------------------------------- Fig 5
def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half) * 100, min(1.0, center + half) * 100


def _mean_ci(values, z=1.96):
    import statistics
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / np.sqrt(len(values))
    return m, z * se


def fig5_best_review():
    """Blind quality judging of each manuscript's four reviews, under two independent judges.

    Panels a-d are the four quality measures; within each panel the two bars per
    reviewer model are the two judge models (grok-4.6 and minimax-m3) scoring the
    same 200 manuscripts, so judge sensitivity is visible inside the main result
    rather than in a separate robustness figure.
    """
    JUDGE_FILES = [("grok", "Judge: Grok-4.6", INK, "results/exp5_review_quality.jsonl"),
                   ("minimax", "Judge: MiniMax-M3", "#7a6bd8", "results/exp5_review_quality_minimax.jsonl")]
    loaded = {}
    for key, _, _, path in JUDGE_FILES:
        loaded[key] = {r["doi"]: r for r in (json.loads(l) for l in open(ROOT / path) if l.strip())}
    dois = sorted(set.intersection(*(set(v) for v in loaded.values())))

    def wilson(k, n):
        if not n:
            return np.nan, 0.0, 0.0
        ph, z = k / n, 1.959964
        d = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / d
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
        return ph * 100, (ph - (c - h)) * 100, ((c + h) - ph) * 100

    def measures(recs, model):
        wins = sum(1 for d in dois if recs[d]["best_review_model"] == model)
        a = wilson(wins, len(dois))
        counts, scores, nit, tot = [], [], 0, 0
        for d in dois:
            r = recs[d]
            letters = {l for l, mm in r["letter_to_model"].items() if mm == model}
            counts.append(sum(1 for c in r["concern_clusters"]
                              if c.get("category") == "relevant" and set(c.get("raised_by", [])) & letters))
            scores += [r["constructiveness"][l] for l in letters if r["constructiveness"].get(l) is not None]
            for c in r["concern_clusters"]:
                if c.get("category") in ("relevant", "nitpicking") and set(c.get("raised_by", [])) & letters:
                    tot += 1
                    nit += c["category"] == "nitpicking"
        counts, scores = np.array(counts, float), np.array(scores, float)
        ci = lambda v: 1.959964 * v.std(ddof=1) / np.sqrt(len(v))
        return dict(a=a, b=(counts.mean(), ci(counts), ci(counts)),
                    c=(scores.mean(), ci(scores), ci(scores)), d=wilson(nit, tot))

    stats_ = {k: {m: measures(loaded[k], m) for m in MODELS} for k, _, _, _ in JUDGE_FILES}

    PANELS = [("a", "judged best (%)", "{:.0f}"),
              ("b", "relevant concerns per manuscript", "{:.1f}"),
              ("c", "constructiveness (1\u20135)", "{:.1f}"),
              ("d", "nitpicking rate (%)", "{:.0f}")]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.7))
    x = np.arange(len(MODELS), dtype=float)
    width = 0.38
    for ax, (letter, ylab, fmt) in zip(axes, PANELS):
        for sign, (key, label, color, _) in zip((-1, 1), JUDGE_FILES):
            vals = np.array([stats_[key][m][letter][0] for m in MODELS])
            lo = np.array([stats_[key][m][letter][1] for m in MODELS])
            hi = np.array([stats_[key][m][letter][2] for m in MODELS])
            pos = x + sign * width / 2
            ax.bar(pos, vals, width=width, color=color, label=label, zorder=3)
            ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.7,
                        capsize=1.6, capthick=0.7, zorder=4)
            for xi, v, h in zip(pos, vals, hi):
                ax.text(xi, v + h + (0.025 * max(vals.max(), 1e-9) + 0.02 * ax.get_ylim()[1]),
                        fmt.format(v), ha="center", va="bottom", fontsize=5.8, color=MUTED, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m].replace(" ", "\n") for m in MODELS],
                           fontsize=6.6, color=INK, linespacing=1.15)
        ax.set_ylabel(ylab, fontsize=7, color=INK)
        if letter == "a":
            ax.set_ylim(0, 105)
        if letter == "c":
            ax.set_ylim(0, 5.4)
            ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.margins(y=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(RULE)
        ax.spines["bottom"].set_color(RULE)
        ax.tick_params(colors=INK, labelsize=6.6, length=2, width=0.6, bottom=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=RULE, linewidth=0.5)
        _panel(ax, letter, x=-0.22, y=1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2,
               frameon=False, fontsize=7, handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / "fig5_best_review.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig5_best_review.png (n=%d manuscripts, two judges)" % len(dois))

def fig6_consensus_quality():
    records = [json.loads(l) for l in open(ROOT / "results" / "exp5_review_quality.jsonl") if l.strip()]
    from collections import Counter

    overlap = Counter()  # (n_reviewers_agreeing, category) -> count
    model_at_level = Counter()  # (n_reviewers_agreeing, model) -> appearances
    for r in records:
        l2m = r["letter_to_model"]
        for c in r["concern_clusters"]:
            level = len(c["raised_by"])
            overlap[(level, c["category"])] += 1
            for letter in c["raised_by"]:
                m = l2m.get(letter)
                if m:
                    model_at_level[(level, m)] += 1

    xs = [1, 2, 3, 4]
    xlabels = ["Raised by\n1 reviewer", "Raised by\n2 reviewers", "Raised by\n3 reviewers", "Raised by\nall 4 reviewers"]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))

    # Panel A: model composition (share of appearances) at each agreement level
    ax = axes[0]
    bottoms = np.zeros(len(xs))
    for m in MODELS:
        totals_at_level = [sum(model_at_level.get((x, mm), 0) for mm in MODELS) for x in xs]
        shares = [model_at_level.get((x, m), 0) / t * 100 if t else 0 for x, t in zip(xs, totals_at_level)]
        ax.bar(xs, shares, bottom=bottoms, width=0.6, color=MODEL_COLOR[m], edgecolor="white", linewidth=0.8,
               label=MODEL_LABEL[m])
        bottoms += np.array(shares)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=6.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of appearances\nat this agreement level", fontsize=7, color=INK)
    ax.set_title("A · Which models make up each\nagreement level", loc="left", fontsize=7.5, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(1.08, 1.32), ncol=4, frameon=False, fontsize=6.3,
              handlelength=1.0, handleheight=1.0, columnspacing=0.9)

    # Panel B: nitpick share by agreement level
    ax = axes[1]
    nitpick_share, totals = [], []
    for x in xs:
        rel = overlap.get((x, "relevant"), 0)
        nit = overlap.get((x, "nitpicking"), 0)
        tot = rel + nit
        totals.append(tot)
        nitpick_share.append(nit / tot * 100 if tot else 0)
    ax.bar(xs, nitpick_share, width=0.55, color=CRITICAL, edgecolor="white", linewidth=1.2, zorder=3)
    for x, v, tot in zip(xs, nitpick_share, totals):
        ax.text(x, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.3, color=INK, fontweight="bold")
        ax.text(x, -3.5, f"n={tot}", ha="center", va="top", fontsize=6, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=6.8)
    ax.set_ylim(-6, max(nitpick_share) * 1.35)
    ax.set_ylabel("Share of concerns judged\nas nitpicking rather than relevant", fontsize=7, color=INK)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_title("B · Consensus tracks quality", loc="left", fontsize=7.5, color=INK)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(bottom=False, colors=INK, labelsize=7)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=RULE, linewidth=0.5)
    axes[0].spines["bottom"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig6_consensus_quality.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig6_consensus_quality.png")


# ---------------------------------------------------------------- Fig 7
def fig7_concern_overlap():
    records = [json.loads(l) for l in open(ROOT / "results" / "exp5_review_quality.jsonl") if l.strip()]
    from itertools import combinations

    own = {m: 0 for m in MODELS}
    pair = {}
    for a, b in combinations(MODELS, 2):
        pair[(a, b)] = 0

    for r in records:
        l2m = r["letter_to_model"]
        for c in r["concern_clusters"]:
            if c["category"] != "relevant":
                continue
            models_here = sorted(set(l2m[l] for l in c["raised_by"] if l in l2m))
            for m in models_here:
                own[m] += 1
            for a, b in combinations(models_here, 2):
                key = (a, b) if (a, b) in pair else (b, a)
                pair[key] += 1

    # row-normalized: cell (i,j) = % of row model i's own relevant concerns
    # that model j also independently raised. Asymmetric by design -- a
    # low-volume model can have nearly all of its concerns "covered" by a
    # high-volume one, while the reverse fraction is small.
    order = MODELS
    n = len(order)
    grid = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                continue
            key = (a, b) if (a, b) in pair else (b, a)
            grid[i, j] = pair[key] / own[a] * 100 if own[a] else 0

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    # sequential magnitude ramp, single hue -- deliberately not GOOD/status-green,
    # since overlap magnitude here is not a "good/bad" judgment
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#f2f0e9", "#2a78d6"])
    cmap.set_bad(color="white")
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = grid[i, j]
            txt_color = "white" if v > 60 else INK
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=txt_color)

    labels = [MODEL_SHORT[m] for m in order]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7.5, color=INK)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.tick_params(left=False, bottom=False, top=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("...also independently raised by this model", fontsize=7, color=INK)
    ax.set_ylabel("Of this model's relevant concerns...", fontsize=7, color=INK)

    own_line = ", ".join(f"{MODEL_SHORT[m]}={own[m]}" for m in order)
    print("own relevant concerns:", own_line)

    fig.tight_layout()
    fig.savefig(OUT / "fig7_concern_overlap.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig7_concern_overlap.png")


def fig8_ngram_fingerprint():
    """Supplementary robustness check for the blind quality-judging result (Fig. 5, S3.3):
    log-odds-ratio bigram/trigram specificity per model, computed directly from raw review
    text with no judge model involved. Negation words (not, no, without...) are kept in the
    stopword-filtered vocabulary, since dropping them silently flips phrases like
    "does not establish" into "does establish"."""
    DATA = {
        "openai/gpt-5.6-sol": {
            "bigrams": [("does not", 12.2), ("not establish", 12.1), ("confidence intervals", 6.9),
                        ("cannot establish", 5.8), ("not demonstrate", 5.5), ("not adequately", 4.6)],
            "trigrams": [("does not establish", 9.8), ("does not demonstrate", 5.3), ("analysis does not", 4.1),
                         ("experiment does not", 3.6), ("does not resolve", 3.5), ("evidence does not", 3.3)],
        },
        "google/gemini-3.7-flash": {
            "bigrams": [("typographical error", 7.6), ("results section", 7.2), ("methods section", 7.0),
                        ("brief sentence", 5.8), ("authors appropriately", 5.3), ("brief discussion", 4.9)],
            "trigrams": [("authors explicitly discuss", 4.1), ("text figure legends", 3.7), ("authors appropriately acknowledge", 3.5),
                         ("family wise error", 3.5), ("correct typographical error", 3.4), ("brief sentence discussion", 3.3)],
        },
        "z-ai/glm-5.3-flash": {
            "bigrams": [("abstract states", 6.5), ("substantially strengthen", 6.2), ("consider reporting", 5.2),
                        ("reference formatting", 4.9), ("fig caption", 4.3), ("tempered accordingly", 3.9)],
            "trigrams": [("main text not", 4.4), ("substantially strengthen claim", 3.3), ("reported main text", 3.2),
                         ("figure legend references", 2.9), ("reference formatting inconsistent", 2.9), ("asserted not demonstrated", 2.8)],
        },
        "deepseek/deepseek-v4-flash-0731": {
            "bigrams": [("not fully", 9.4), ("not provide", 7.8), ("cost effectiveness", 7.8),
                        ("cost effective", 6.8), ("authors state", 6.2), ("authors not", 6.2)],
            "trigrams": [("paper does not", 5.6), ("not fully described", 4.2), ("authors discuss potential", 3.3),
                         ("authors acknowledge limitation", 3.2), ("authors not provide", 3.1), ("authors not report", 2.7)],
        },
    }
    order = MODELS

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    for letter, (ax, m) in zip("abcd", zip(axes.flat, order)):
        bi = [(p, z, "2-gram") for p, z in DATA[m]["bigrams"]]
        tri = [(p, z, "3-gram") for p, z in DATA[m]["trigrams"]]
        combined = sorted(bi + tri, key=lambda r: -r[1])[:10]  # merged, ranked purely by z, not by n-gram length
        rows = [(f"{p}  ({kind})", z) for p, z, kind in combined]
        rows = rows[::-1]  # so highest-z phrase plots at top after barh's bottom-up order
        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]
        color = MODEL_COLOR[m]
        ax.barh(range(len(rows)), values, color=color, height=0.62)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=7.6, color=INK)
        ax.set_xlabel("log-odds z-score vs. other three models", fontsize=7.5, color=MUTED)
        _panel(ax, letter, x=-0.30, y=1.02)
        ax.text(0.0, 1.02, MODEL_LABEL[m], transform=ax.transAxes, fontsize=9.5,
                color=INK, ha="left", va="bottom")
        ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        ax.axvline(0, color=RULE, linewidth=0.8)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "fig8_ngram_fingerprint.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig8_ngram_fingerprint.png")


# ---------------------------------------------------------------- Fig 9
def fig9_second_round():
    """Effect of framing the same manuscript as a post-first-round revision.

    Panel a: mean recommendation severity per model under the two framings
    (paired within manuscript). Panel b: where the decision mass moves, pooled.
    """
    def load(name):
        with open(ROOT / "results" / name) as f:
            rows = [r for r in csv.DictReader(f) if r["model"] in MODELS]
        seen, out = set(), {}
        for r in rows:                       # first occurrence wins
            k = (r["doi"], r["model"])
            if k in seen:
                continue
            seen.add(k)
            out[k] = r["decision"]
        return out

    first, second = load("exp1_decisions.csv"), load("exp1_secondround.csv")
    keys = [k for k in first if k in second and first[k] in SEV and second[k] in SEV]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # --- a: paired mean severity, first round -> presented as a revision
    ax = axes[0]
    y = np.arange(len(MODELS))[::-1]
    for yi, m in zip(y, MODELS):
        ks = [k for k in keys if k[1] == m]
        a = np.mean([SEV[first[k]] + 1 for k in ks])
        b = np.mean([SEV[second[k]] + 1 for k in ks])
        ax.annotate("", xy=(b, yi), xytext=(a, yi),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.34",
                                    color=MUTED, linewidth=1.1, shrinkA=2.5, shrinkB=0))
        ax.plot([a], [yi], "o", ms=5.5, mfc="white", mec=INK, mew=1.1, zorder=3)
        ax.plot([b], [yi], "o", ms=5.5, color=CRITICAL, zorder=3)
        # label from the exact paired shift (integer sum / n), rounded half-up, so it matches the text;
        # b - a carries float error that can round -0.655 to -0.65
        shift = Decimal(sum(SEV[second[k]] - SEV[first[k]] for k in ks)) / len(ks)
        ax.text(b - 0.06, yi + 0.26, f"{shift.quantize(Decimal('0.01'), ROUND_HALF_UP):+}", fontsize=6.3, color=MUTED, ha="right", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=7.5, color=INK)
    ax.set_xlim(1.0, 3.45)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Accept", "Minor\nrevision", "Major\nrevision"], fontsize=7)
    ax.set_xlabel("Mean recommendation", fontsize=7.5, color=INK, labelpad=2)
    ax.set_ylim(-0.45, len(MODELS) - 0.4)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(left=False, colors=INK, labelsize=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5)
    h = [plt.Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec=INK, mew=1.1),
         plt.Line2D([], [], marker="o", ls="", ms=5.5, color=CRITICAL)]
    ax.legend(h, ["First-round framing", "Presented as a revision"], loc="upper center",
              bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False, fontsize=6.5,
              handletextpad=0.3, columnspacing=1.2)
    _panel(ax, "a")

    # --- b: pooled decision mix under each framing
    ax = axes[1]
    yb = np.array([1.0, 0.0])
    for framing, yi in zip((first, second), yb):
        left = 0.0
        for i, key in enumerate(SEV_KEYS):
            v = 100 * sum(1 for k in keys if framing[k] == key) / len(keys)
            ax.barh([yi], [v], left=left, height=0.5, color=SEV_COLORS[i],
                    edgecolor="white", linewidth=1.2)
            if v >= 6:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white" if i in (0, 3) else INK)
            left += v
    ax.set_yticks(yb)
    ax.set_yticklabels(["First-round\nframing", "Presented as\na revision"], fontsize=7.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7)
    ax.set_xlabel("Share of reviews", fontsize=7.5, color=INK, labelpad=2)
    ax.set_ylim(-0.55, 1.55)
    for s in ("left", "bottom", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False, colors=INK, labelsize=7)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEV_COLORS]
    ax.legend(handles, ["Accept", "Minor revision", "Major revision", "Reject"],
              loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False,
              fontsize=6.5, handlelength=1.0, handleheight=1.0, columnspacing=1.0)
    _panel(ax, "b")

    fig.tight_layout(w_pad=2.4)
    fig.savefig(OUT / "fig9_second_round.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig9_second_round.png (n={len(keys)} paired reviews)")


if __name__ == "__main__":
    fig1_status_by_model()
    fig2_status_all_papers()
    fig4_exp2_detection(by_category=False)  # the manuscript shows categories pooled
    fig5_best_review()
    fig7_concern_overlap()
    fig8_ngram_fingerprint()
    fig9_second_round()


# ---------------------------------------------------------------- Study design
def fig_study_design():
    """Schematic of the corpus, the reviewer panel, and the three experiments."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig = plt.figure(figsize=(7.2, 4.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    LH = 0.0265  # body line height in axes fraction

    def box(x, y, w, h, title, body, first_line_y=None, title_size=8.2, body_size=6.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor="white", edgecolor=RULE, linewidth=0.9, zorder=2))
        ty = y + h - 0.022
        if title:
            ax.text(x + 0.018, ty, title, ha="left", va="top", fontsize=title_size, color=INK,
                    weight="bold", linespacing=1.25, zorder=3)
            ty -= 0.043 * (title.count("\n") + 1)
        if first_line_y is not None:
            ty = first_line_y
        for line in body:
            if line:
                ax.text(x + 0.018, ty, line, ha="left", va="top", fontsize=body_size,
                        color=INK if line.startswith("\u2192") else MUTED, zorder=3)
            ty -= LH

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color=INK, shrinkA=0, shrinkB=0, zorder=4))

    # --- corpus
    box(0.045, 0.845, 0.91, 0.135, "Corpus",
        ["200 research articles \u00b7 20 general-science, life-science and clinical-medicine journals",
         "all published in 2026 under CC BY 4.0 \u00b7 full text supplied to the model, which was not",
         "told the article had already been published"])
    arrow(0.5, 0.845, 0.5, 0.812)

    # --- reviewer panel
    box(0.045, 0.615, 0.91, 0.197, "Reviewer panel \u2014 blinded protocol",
        ["identical system prompt (Appendix A) naming the manuscript's true target journal; each model",
         "asked for the review it would submit to the editor: a summary, major and minor concerns, and",
         "one of accept / minor revision / major revision / reject"])
    mx = 0.062
    for m in MODELS:
        ax.add_patch(FancyBboxPatch((mx, 0.628), 0.212, 0.033,
                                    boxstyle="round,pad=0.003,rounding_size=0.008",
                                    facecolor=MODEL_COLOR[m], edgecolor="none", zorder=3))
        ax.text(mx + 0.106, 0.6445, MODEL_LABEL[m], ha="center", va="center", fontsize=7.0,
                color="white", weight="bold", zorder=4)
        mx += 0.2245

    # --- fan-out to the three experiment lanes
    ax.plot([0.19, 0.81], [0.578, 0.578], color=INK, linewidth=0.8, zorder=1)
    arrow(0.5, 0.615, 0.5, 0.579)
    for x in (0.19, 0.5, 0.81):
        arrow(x, 0.578, x, 0.545)

    lanes = [
        (0.045, "Experiment 1\nBaseline review",
         ["200 manuscripts \u00d7 4 models,", "unmodified text = 800 reviews", "",
          "every manuscript re-reviewed a", "second time under the identical",
          "protocol (800 further reviews)", "",
          "\u2192 recommendation distribution", "\u2192 intra- vs inter-model agreement"]),
        (0.355, "Experiment 2\nError injection",
         ["one severe, field-specific flaw", "planted per manuscript by an off-",
          "panel model (Grok 4.6) as a verbatim", "string substitution; statistical or",
          "methodological, round-robin", "",
          "800 reviews of the doctored text", "",
          "\u2192 detection judged by off-panel", "    Claude Opus 5 (yes / partial / no)"]),
        (0.665, "Experiment 3\nReview quality",
         ["the four Experiment-1 reviews of", "each manuscript, relabelled Review",
          "A\u2013D in a per-manuscript random", "order and presented as anonymous",
          "referee reports", "",
          "200 judgings by an off-panel judge", "(Grok 4.6)", "",
          "\u2192 concern clusters: relevant vs", "    nitpicking; constructiveness 1\u20135;",
          "    single best review"]),
    ]
    for x, title, body in lanes:
        box(x, 0.145, 0.29, 0.40, title, body, first_line_y=0.455, body_size=6.6)

    # --- secondary analyses
    box(0.045, 0.018, 0.91, 0.105, "Secondary analyses on the same reviews",
        ["phrase-specificity log-odds fingerprint of each model's review vocabulary  \u00b7  training-data",
         "recognition probe (can a model name the manuscript it has just reviewed?)"])

    fig.savefig(OUT / "fig_study_design.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_study_design.png")
