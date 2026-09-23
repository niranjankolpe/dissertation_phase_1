#!/usr/bin/env python3
"""Matched-size (3 vs 3) subgroup instability check - removes the 4-vs-8
feature-count confound from 4_seed_instability.py.

Group A (leaky, correlated):   AC Ir, AC Iy, AC Ib  -- 3 near-identical
    phase currents, all correlated with the winner (DC Current) and each
    other. Tests the first-mover-bias mechanism directly (arXiv 2603.22346,
    2605.21492): does rank among CORRELATED runners-up scramble across seeds?
Group B (leak_aware, control):  MODULE_TEMP, Amb_Temp, WIND_Speed -- 3
    weather runners-up behind IRR, weaker mutual correlation. Same group
    size, same "runner-up" role, but not the tight collinear cluster.

If A is meaningfully less stable than B despite equal group size, that is
clean (dimensionality-matched) evidence for the reframed thesis question.
If A and B look similar, the earlier signal was likely just the small-n
Spearman noise flagged in the previous review note.

Run: python 6_subgroup_instability.py [--importance results/importance.csv]
Output: analysis_notes/<timestamp>_subgroup_instability.txt
        analysis_notes/<timestamp>_subgroup_instability.csv"""
import argparse, datetime, itertools, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

GROUP_A = ("leaky", ["AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps"])
GROUP_B = ("leak_aware", ["MODULE_TEMP", "Amb_Temp", "WIND_Speed"])


def cv(x):
    x = np.asarray(x, float)
    m = np.mean(x)
    return float(np.std(x) / m) if abs(m) > 1e-12 else float("nan")


def analyze(df, label, fs, cols):
    rows_rank, rows_cv = [], []
    for method, g in df[df.featureset == fs].groupby("method"):
        g = g.set_index("seed")[cols]
        seeds = sorted(g.index.tolist())
        for s1, s2 in itertools.combinations(seeds, 2):
            a, b = np.abs(g.loc[s1].values), np.abs(g.loc[s2].values)
            rho = spearmanr(a, b).correlation
            top1 = float(np.argmax(a) == np.argmax(b))
            rows_rank.append({"group": label, "featureset": fs, "method": method,
                               "seed_pair": f"{s1}-{s2}",
                               "spearman": 0.0 if rho is None or np.isnan(rho) else rho,
                               "top1_of_group_match": top1})
        for feat in cols:
            rows_cv.append({"group": label, "featureset": fs, "method": method,
                             "feature": feat, "coeff_of_variation": cv(g[feat].values)})
    return rows_rank, rows_cv


def main(path):
    df = pd.read_csv(path)
    rank_rows, cv_rows = [], []
    for label, (fs, cols) in (("A_correlated_leaky", GROUP_A), ("B_control_leak_aware", GROUP_B)):
        r, c = analyze(df, label, fs, cols)
        rank_rows += r; cv_rows += c

    rank_df, cv_df = pd.DataFrame(rank_rows), pd.DataFrame(cv_rows)
    rank_summary = rank_df.groupby(["group", "method"])[["spearman", "top1_of_group_match"]].mean().round(3)
    cv_summary = cv_df.groupby(["group", "method"])["coeff_of_variation"].mean().round(3)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs("analysis_notes", exist_ok=True)
    txt_path = f"analysis_notes/{ts}_subgroup_instability.txt"
    csv_path = f"analysis_notes/{ts}_subgroup_instability.csv"
    pd.concat([rank_df, cv_df.rename(columns={"feature": "extra_feature"})], axis=0, ignore_index=True, sort=False) \
        .to_csv(csv_path, index=False)

    lines = [
        f"[{ts}] Subgroup instability (matched 3-vs-3, removes feature-count confound)",
        f"Source: {path}",
        "",
        "Group A = leaky: AC Ir/Iy/Ib (correlated phase currents, runners-up behind DC Current)",
        "Group B = leak_aware: MODULE_TEMP/Amb_Temp/WIND_Speed (control runners-up behind IRR)",
        "",
        "RANK STABILITY WITHIN GROUP (spearman + top1-of-3 match across seed pairs; lower = less stable):",
        rank_summary.to_string(),
        "",
        "MAGNITUDE INSTABILITY (coefficient of variation of each feature's attribution share across seeds; higher = less stable):",
        cv_summary.to_string(),
        "",
        "READ: if group A spearman/top1 << group B at matched n=3, and/or A's CV >> B's CV,",
        "that is dimensionality-matched evidence that the correlated cluster is the source of",
        "instability (supports arXiv 2603.22346 / 2605.21492 extending to this Transformer).",
        "If A and B are close, the effect is weak/unconfirmed at n=3 seeds - say so plainly.",
    ]
    open(txt_path, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten: {txt_path}\nwritten: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--importance", default="results/importance.csv")
    a = p.parse_args()
    main(a.importance)
