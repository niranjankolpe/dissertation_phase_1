#!/usr/bin/env python3
"""Cross-seed attribution instability (first-mover-bias-style check) on already-run results.
No retraining. Reads results/importance.csv (3 seeds already present).
Run: python 4_seed_instability.py
Output: results/seed_instability.csv + results/SEED_INSTABILITY_REPORT.txt"""
import itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr

FEAT_COLS = ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)",
             "DC Current in Amps", "AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps"]

df = pd.read_csv("results/importance.csv")
rows = []
for (fs, method), g in df.groupby(["featureset", "method"]):
    cols = [c for c in FEAT_COLS if g[c].notna().all()]
    g = g.set_index("seed")[cols]
    for s1, s2 in itertools.combinations(g.index, 2):
        a, b = np.abs(g.loc[s1].values), np.abs(g.loc[s2].values)
        rho = spearmanr(a, b).correlation
        top2 = len(set(np.argsort(a)[-2:]) & set(np.argsort(b)[-2:])) / 2
        rows.append({"featureset": fs, "method": method, "seed_pair": f"{s1}-{s2}",
                     "spearman": 0.0 if np.isnan(rho) else rho,
                     "top1_match": float(np.argmax(a) == np.argmax(b)), "top2_overlap": top2})

out = pd.DataFrame(rows)
out.to_csv("results/seed_instability.csv", index=False)
summary = out.groupby(["featureset", "method"])[["spearman", "top1_match", "top2_overlap"]].mean().round(3)
print(summary)
open("results/SEED_INSTABILITY_REPORT.txt", "w").write(
    "Cross-seed attribution instability (lower spearman/top1/top2 = more unstable)\n\n" + summary.to_string())
