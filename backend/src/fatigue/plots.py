"""Static figures for the results folder (matplotlib, Agg). Palette: validated categorical slots 1-3."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _ax(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    ax.set_facecolor("#fcfcfb"); fig.patch.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def bar(names, values, path, title, ylabel, colors=None, errs=None, annotate="{:.2f}", ylim=None, note=None):
    fig, ax = _ax((max(5.5, 1.1 * len(names)), 4))
    x = np.arange(len(names))
    ax.bar(x, values, width=0.55, color=colors or BLUE, yerr=errs, ecolor=MUTED, capsize=3)
    for xi, v in zip(x, values):
        if np.isfinite(v):
            ax.text(xi, v, annotate.format(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9); ax.set_title(title, loc="left", fontsize=10, color=INK)
    if ylim: ax.set_ylim(*ylim)
    if note: fig.text(0.01, 0.005, note, fontsize=7, color=MUTED)
    _save(fig, path)


def per_subject_dots(tables: dict, path, title, note=None):
    """tables: {method: per-subject DataFrame with columns subject, acc}."""
    fig, ax = _ax((7, 4.2))
    cols = [ORANGE, BLUE, AQUA, MUTED]
    for i, (name, t) in enumerate(tables.items()):
        y = np.full(len(t), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(t))
        ax.scatter(t["acc"], y, s=26, color=cols[i % 4], alpha=0.85, edgecolor="#fcfcfb", lw=0.8)
        ax.plot([t["acc"].mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.6)
    ax.set_yticks(range(len(tables))); ax.set_yticklabels(list(tables), fontsize=8)
    ax.set_xlabel("accuracy on the held-out person (black bar = mean)", color=MUTED, fontsize=9)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    if note: fig.text(0.01, 0.005, note, fontsize=7, color=MUTED)
    _save(fig, path)


def lead_strip(leads: dict, path, title, note=None):
    fig, ax = _ax((7, 4))
    cols = [ORANGE, BLUE, AQUA, MUTED]
    for i, (name, v) in enumerate(leads.items()):
        v = np.asarray(v, float)
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v,
                   s=26, color=cols[i % 4], alpha=0.85, edgecolor="#fcfcfb", lw=0.8)
        if len(v):
            ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.6)
    ax.set_xticks(range(len(leads))); ax.set_xticklabels(list(leads), fontsize=8)
    ax.set_ylabel("seconds of warning before first microsleep", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    if note: fig.text(0.01, 0.005, note, fontsize=7, color=MUTED)
    _save(fig, path)


def fairness_bars(df: pd.DataFrame, path, title, note=None):
    """Equalised-odds gap per method x attribute, with subject-bootstrap interval where estimable."""
    g = df[df["row"] == "GAP (1 minus 0)"].reset_index(drop=True)
    names = {"fixed EAR baseline": "fixed EAR", "ours (before correction)": "ours, before"}
    labels = []
    for r in g.itertuples():
        m = names.get(r.method, "ours, after" if "after" in r.method else r.method)
        labels.append(f"{m}\n{r.attr}")
    fig, ax = _ax((max(7, 1.2 * len(g)), 4.4))
    vals = g["eq_odds_gap"].to_numpy(float)
    lo, hi = g["eq_odds_ci_lo"].to_numpy(float), g["eq_odds_ci_hi"].to_numpy(float)
    x = np.arange(len(g))
    ax.bar(x, vals, width=0.5, color=BLUE)
    for i, v in enumerate(vals):
        if np.isfinite(lo[i]):
            ax.errorbar(i, v, yerr=[[max(v - lo[i], 0)], [max(hi[i] - v, 0)]], fmt="none", ecolor=MUTED, capsize=3)
        else:
            ax.text(i, v, "no interval\n(group of 1)", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("equalised-odds gap", color=MUTED, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    if note: fig.text(0.01, 0.005, note, fontsize=7, color=MUTED)
    _save(fig, path)
