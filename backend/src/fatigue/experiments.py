"""The full experiment suite -> results/ (CSV, JSON, PNG, RESULTS.md)."""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import fairness as fr
from . import plots, schema
from .config import Config
from .windows import KEY, build_windows, enrol_profile


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c, v in zip(cols, r):
            if isinstance(v, (float, np.floating)):
                v = float(v)
                cells.append("" if not np.isfinite(v) else
                             (str(int(v)) if c in ("n_subjects", "n_clips", "people") else floatfmt.format(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _row(name: str, pred: pd.DataFrame, hit="hit") -> Dict:
    s = ev.summarize(pred, hit)
    return {"method": name, "acc (pooled)": s["pooled_acc"], "balanced acc": s["pooled_bal_acc"],
            "TPR": s["pooled_tpr"], "FPR": s["pooled_fpr"], "alert rate": s["pooled_alert_rate"],
            "per-person acc mean": s["subj_acc_mean"], "per-person acc std": s["subj_acc_std"],
            "people": s["n_subjects"]}


def run_all(frames: pd.DataFrame, meta: Optional[pd.DataFrame], cfg: Config, outdir: str,
            synthetic: bool = False, n_jobs: int = 1, log=print) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    cfg_z = cfg.replace(model={"use_z": True})
    seed = cfg.evaluation.seed
    R: Dict = {"synthetic": synthetic, "config_hash": cfg.hash()}

    log("[1/7] building windows (adaptive, fixed, enrolled) ...")
    Wa, E = build_windows(frames, cfg, "adaptive", n_jobs=n_jobs)
    Wf, _ = build_windows(frames, cfg, "fixed", n_jobs=n_jobs)
    We, _ = build_windows(frames, cfg, "enrolled", n_jobs=n_jobs)
    # enrol on a chosen clip of the same person: alert clip (normal) or drowsy clip (tired start)
    prof_alert, prof_drowsy = {}, {}
    for s, g in frames.groupby("subject"):
        for lab, store in ((cfg.evaluation.neg_labels[0], prof_alert), (cfg.evaluation.pos_labels[0], prof_drowsy)):
            clip = g[g["label"] == lab]
            if not clip.empty:
                store[s] = enrol_profile(clip.sort_values("t"), cfg)
    mk = lambda store: {(s, v): store[s] for s, v in frames[["subject", "video"]].drop_duplicates().itertuples(index=False)
                        if s in store}
    Wea, _ = build_windows(frames, cfg, "enrolled", profiles=mk(prof_alert), n_jobs=n_jobs)
    Wed, _ = build_windows(frames, cfg, "enrolled", profiles=mk(prof_drowsy), n_jobs=n_jobs)
    Wa.to_csv(os.path.join(outdir, "windows_adaptive.csv"), index=False)

    log("[2/7] main comparison (leave-one-person-out) ...")
    Da = ev.prepare(Wa, cfg)
    base = ev.run_loso_baseline(Da, cfg)
    pop = ev.run_loso(Wf, cfg, "fixed")
    ours = ev.run_loso(Da, cfg, "adaptive", prepared=True)
    ours_z = ev.run_loso(Wa, cfg_z, "adaptive")
    enr_self = ev.run_loso(We, cfg_z, "enrolled")
    enr_alert = ev.run_loso(Wea, cfg_z, "enrolled")
    enr_drowsy = ev.run_loso(Wed, cfg_z, "enrolled")
    main = pd.DataFrame([
        _row("A. fixed EAR threshold (baseline)", base),
        _row("B. same model, population reference (no per-driver adaptation)", pop),
        _row("C. OURS: adaptive per-driver reference", ours),
        _row("D. adaptive + early signals z-scored", ours_z),
        _row("E. enrolled on own first 60 s + z", enr_self),
        _row("F. enrolled on the person's ALERT clip + z", enr_alert),
        _row("G. enrolled on the person's DROWSY clip + z (tired start)", enr_drowsy),
    ])
    main.to_csv(os.path.join(outdir, "main_comparison.csv"), index=False)

    log("[3/7] wrong-split demonstration (leakage) ...")
    leaky = ev.run_leaky(Da, cfg, prepared=True, seed=seed)
    leak = pd.DataFrame([_row("random window split (WRONG: same person in train and test)", leaky),
                         _row("leave-one-person-out (correct)", ours)])
    leak.to_csv(os.path.join(outdir, "leakage_demo.csv"), index=False)

    log("[4/7] ablations ...")
    abl_rows = []
    for name, cols in schema.ABLATIONS.items():
        p = ours if name == "full" else ev.run_loso(Da, cfg, "adaptive", cols=cols, prepared=True)
        abl_rows.append(_row(name, p))
    cfg_ms = cfg_z.replace(reference={"norm_method": "mean_std"})
    Wms, _ = build_windows(frames, cfg_ms, "adaptive", n_jobs=n_jobs)
    abl_rows.append(_row("z-scores: median/IQR (adaptive+z)", ours_z))
    abl_rows.append(_row("z-scores: mean/std (adaptive+z)", ev.run_loso(Wms, cfg_ms, "adaptive")))
    for L in cfg.evaluation.window_lengths_s:
        if abs(L - cfg.window.length_s) < 1e-9:
            abl_rows.append(_row(f"window {L:g} s", ours)); continue
        cfgL = cfg.replace(window={"length_s": L})
        WL, _ = build_windows(frames, cfgL, "adaptive", window_s=L, n_jobs=n_jobs)
        abl_rows.append(_row(f"window {L:g} s", ev.run_loso(WL, cfgL, "adaptive")))
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(os.path.join(outdir, "ablations.csv"), index=False)

    log("[5/7] alerts: earliness and false alarms per hour ...")
    ours_l = ev.add_levels(ours, cfg, True)
    ours_nl = ev.add_levels(ours, cfg, False)
    pop_l = ev.add_levels(pop, cfg, True)
    base_s = ev.add_baseline_state(base, cfg, True)
    base_ns = ev.add_baseline_state(base, cfg, False)
    early_rows = []
    for name, p, col, lvl in [("fixed EAR baseline", base_s, "state", 1),
                              ("population reference", pop_l, "level", 1),
                              ("OURS first alert (MILD+)", ours_l, "level", 1),
                              ("OURS MODERATE+", ours_l, "level", 2),
                              ("OURS SEVERE", ours_l, "level", 3)]:
        early_rows.append({"method": name, **ev.earliness(p, E, cfg, col, lvl)})
    early = pd.DataFrame(early_rows)
    early.to_csv(os.path.join(outdir, "earliness.csv"), index=False)
    fa = pd.DataFrame([
        {"method": "fixed EAR baseline", "with persistence": ev.false_alarms_per_hour(base_s, cfg, "state"),
         "without persistence": ev.false_alarms_per_hour(base_ns, cfg, "state")},
        {"method": "OURS (any alert)", "with persistence": ev.false_alarms_per_hour(ours_l, cfg, "level"),
         "without persistence": ev.false_alarms_per_hour(ours_nl, cfg, "level")}])
    fa.to_csv(os.path.join(outdir, "false_alarms_per_hour.csv"), index=False)
    lead_raw = {}
    for name, p, col in [("fixed EAR", base_s, "state"), ("population", pop_l, "level"), ("ours", ours_l, "level")]:
        leads = []
        for k, g in p[p["label"].isin(cfg.evaluation.earliness_labels)].sort_values("t_end").groupby(KEY, sort=False):
            e = E[(E["subject"] == k[0]) & (E["label"] == k[1]) & (E["video"] == k[2]) & (E["t"] >= cfg.evaluation.warmup_s)]
            if e.empty:
                continue
            tms = float(e["t"].min()); a = g[(g[col] >= 1) & (g["t_end"] <= tms)]
            leads.append(tms - float(a["t_end"].iloc[0]) if not a.empty else 0.0)
        lead_raw[name] = leads

    log("[6/7] fairness audit ...")
    fair = pd.DataFrame()
    if meta is not None:
        attrs = [a for a in ("glasses", "facial_hair") if a in meta.columns]
        preds = {"fixed EAR baseline": (base, "hit"), "ours (before correction)": (ours, "hit")}
        for a in attrs:
            gm = dict(zip(meta["subject"], meta[a].astype(int)))
            pc = ev.run_loso(Da, cfg, "adaptive", group_map=gm, prepared=True)
            preds[f"ours (after correction: {a})"] = (pc, "hit_group")
        # report each correction only against its own attribute
        parts = []
        for name, (p, col) in preds.items():
            for a in attrs:
                if "after correction" in name and a not in name:
                    continue
                parts.append(fr.fairness_report({name: (p, col)}, meta, [a], cfg.evaluation.n_boot, seed))
        fair = pd.concat(parts, ignore_index=True)
        fair.to_csv(os.path.join(outdir, "fairness.csv"), index=False)

    log("[7/7] figures and report ...")
    note = "SYNTHETIC DATA - shows the code works, not real-driver performance." if synthetic else None
    plots.bar(["random window split\n(wrong)", "leave-one-person-out\n(correct)"],
              [leak.iloc[0]["acc (pooled)"], leak.iloc[1]["acc (pooled)"]],
              os.path.join(outdir, "fig_leakage.png"), "Same model, two ways of splitting data",
              "accuracy", colors=[plots.ORANGE, plots.BLUE], ylim=(0, 1.05), note=note)
    plots.per_subject_dots({"fixed EAR": ev.per_subject(base), "population ref": ev.per_subject(pop),
                            "ours": ev.per_subject(ours)}, os.path.join(outdir, "fig_per_person.png"),
                           "Accuracy on each held-out person", note=note)
    plots.lead_strip(lead_raw, os.path.join(outdir, "fig_earliness.png"),
                     "Warning time before first microsleep", note=note)
    plots.bar(list(abl["method"]), list(abl["balanced acc"]), os.path.join(outdir, "fig_ablations.png"),
              "Ablations (balanced accuracy, leave-one-person-out)", "balanced accuracy",
              ylim=(0, 1.05), note=note)
    if len(fair):
        plots.fairness_bars(fair, os.path.join(outdir, "fig_fairness.png"),
                            "Equalised-odds gap by subgroup (error bars: 95% subject bootstrap)", note=note)

    R.update({"main": main.to_dict("records"), "leakage": leak.to_dict("records"),
              "earliness": early.to_dict("records"), "false_alarms_per_hour": fa.to_dict("records")})
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else str(o))
    _write_report(outdir, cfg, synthetic, main, leak, abl, early, fa, fair, frames)
    log(f"done -> {outdir}")
    return R


def _write_report(outdir, cfg, synthetic, main, leak, abl, early, fa, fair, frames):
    n_sub = frames["subject"].nunique()
    L = ["# Results", ""]
    if synthetic:
        L += ["> **SYNTHETIC DATA.** These numbers come from a simulator. They show the pipeline runs end to end "
              "and that its checks behave; they say nothing about real drivers. Re-run on RLDD for real numbers.", ""]
    L += [f"People: {n_sub}. Protocol: leave-one-person-out. Config hash: `{cfg.hash()}`.",
          "Windows scored only after a "
          f"{cfg.evaluation.warmup_s:g} s warm-up. Classes: alert (label 0) vs drowsy (label 10).", "",
          "## 1. Main comparison", _md(main), "",
          "## 2. Why the split matters", _md(leak), "",
          "## 3. Ablations", _md(abl), "",
          f"## 4. Earliness (microsleep = continuous eye closure of at least {cfg.eyes.microsleep_s:g} s; "
          "this threshold is an assumption)", _md(early), "",
          "## 5. False alarms per hour of ALERT driving", _md(fa), ""]
    if len(fair):
        L += ["## 6. Fairness slice (glasses, facial hair)",
              "Group 1 minus group 0. Cells are tiny: a methodology demonstration, not a certification. "
              "No gender or ethnicity is inferred.", _md(fair), ""]
    L += ["## Limits to state first",
          "- RLDD drowsiness is *acted* by volunteers, not real sleep deprivation.",
          "- Few people: every number has wide uncertainty. Report per-person spread, not just the pooled number.",
          "- Compensatory-behaviour detectors have no ground-truth labels; they are judged only by the ablation.",
          "- Within one 10-minute clip a driver is either alert or drowsy, so a per-driver history cannot show "
          "*change*; that effect is only visible in the progressive/shift data.",
          "- Microsleep threshold for the earliness metric is configurable (`eyes.microsleep_s`)."]
    with open(os.path.join(outdir, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(L))
