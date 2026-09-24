#!/usr/bin/env python3
"""THE decisive check: does the "leakage helps cross-site transfer more than
in-site" pattern (Pillar 2) hold on a THIRD independent site, or was it just
a coincidence of these particular 2 plants? Cheap HistGBDT row-level check
(same method as 8_cross_site_quick.py), now 3-way.

Uses the REDUCED 3-feature weather set (Amb_Temp, WIND_Speed, IRR) for ALL
THREE plants, because Plant C has no module-temperature channel - dropping
MODULE_TEMP from A and B too keeps the comparison fair (same feature count/
type everywhere), at the cost of not being directly numerically comparable
to the earlier 4-feature A vs B results in results_cross_site*/.

Leaky feature counts differ slightly by site (A/B have 4 current channels:
DC + 3 AC phases; C has 3: only 3 AC phases, no DC) - noted, not hidden.

Run: python 13_three_site_quick.py
Output: analysis_notes/<ts>_three_site_quick.{txt,csv}"""
import datetime, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

WEATHER3 = ["Amb_Temp", "WIND_Speed", "IRR (W/m2)"]
TARGET = "AC Power in Watts"
SEED = 42


def load_A():
    df = pd.read_csv("Generation_data.csv").dropna().reset_index(drop=True)
    df["WIND_Speed"] = df["WIND_Speed"] / 100.0
    # 3 leak features for a fair 3-vs-3-vs-3 comparison with Plant C (which has no DC channel)
    df["leak1"], df["leak2"], df["leak3"] = df["AC Ir in Amps"], df["AC Iy in Amps"], df["AC Ib in Amps"]
    return df, WEATHER3 + ["leak1", "leak2", "leak3"]


def load_B():
    df = pd.read_csv("data/plantB_nist_ground_2017.csv").dropna().reset_index(drop=True)
    df["leak1"], df["leak2"], df["leak3"] = df["AC Ir in Amps"], df["AC Iy in Amps"], df["AC Ib in Amps"]
    return df, WEATHER3 + ["leak1", "leak2", "leak3"]


def load_C():
    df = pd.read_csv("data/plantC_hk_lsk_north.csv").dropna().reset_index(drop=True)
    df["leak1"], df["leak2"], df["leak3"] = df["AC_1 in Amps"], df["AC_2 in Amps"], df["AC_3 in Amps"]
    return df, WEATHER3 + ["leak1", "leak2", "leak3"]


def fit_eval(train_df, test_df, feats, same_site, cap_ref):
    Xtr, ytr = train_df[feats].values, train_df[TARGET].values
    if same_site:
        Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, test_size=0.3, random_state=SEED)
    else:
        Xte, yte = test_df[feats].values, test_df[TARGET].values
    m = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr, ytr)
    pred = m.predict(Xte)
    return r2_score(yte, pred), mean_squared_error(yte, pred) ** 0.5, mean_squared_error(yte, pred) ** 0.5 / cap_ref


def main():
    plants = {"A": load_A(), "B": load_B(), "C": load_C()}
    caps = {k: np.percentile(df[TARGET], 99.5) for k, (df, _) in plants.items()}
    print("rows:", {k: len(df) for k, (df, _) in plants.items()}, flush=True)
    print("leaky feature counts:", {k: len(feats) for k, (_, feats) in plants.items()}, flush=True)

    rows = []
    for train_name, (train_df, feats) in plants.items():
        for test_name, (test_df, _) in plants.items():
            same = train_name == test_name
            for fs, use_feats in (("leak_aware", WEATHER3), ("leaky", feats)):
                r2, rmse, nrmse = fit_eval(train_df, test_df, use_feats, same, caps[test_name])
                rows.append({"train": train_name, "test": test_name, "featureset": fs,
                             "same_site": same, "R2": r2, "RMSE": rmse, "RMSE_over_cap": nrmse})
                print(f"{train_name}->{test_name:1s} {fs:11s} R2={r2:8.4f} RMSE/cap={nrmse:.4f}", flush=True)

    out = pd.DataFrame(rows)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs("analysis_notes", exist_ok=True)
    csv_path = f"analysis_notes/{ts}_three_site_quick.csv"
    txt_path = f"analysis_notes/{ts}_three_site_quick.txt"
    out.to_csv(csv_path, index=False)

    piv = out.pivot_table(index=["train", "test"], columns="featureset", values="R2").round(4)
    piv["gap"] = (piv["leaky"] - piv["leak_aware"]).round(4)
    cross_gap = piv[piv.index.get_level_values(0) != piv.index.get_level_values(1)]["gap"].mean()
    same_gap = piv[piv.index.get_level_values(0) == piv.index.get_level_values(1)]["gap"].mean()

    lines = [f"[{ts}] THREE-SITE cross-site quick check (HistGBDT, 3-feature weather set)",
             f"Plants: A=Kaggle India ({len(plants['A'][0])} rows), B=NIST USA ({len(plants['B'][0])} rows), "
             f"C=HKUST Hong Kong ({len(plants['C'][0])} rows)",
             "", piv.to_string(), "",
             f"Mean leaky-vs-leak_aware R2 gap, IN-SITE (train==test): {same_gap:.4f}",
             f"Mean leaky-vs-leak_aware R2 gap, CROSS-SITE (train!=test): {cross_gap:.4f}",
             f"Ratio (cross-site gap / in-site gap): {cross_gap/same_gap:.2f}x" if same_gap else "n/a",
             "",
             "READ: Pillar 2 claimed cross-site gap >> in-site gap on 2 sites (A,B). This is the",
             "3rd-site test. If the ratio above is still clearly >1 with C included (6 cross-site",
             "pairs now, not just 2), that is much stronger evidence this is a real pattern, not a",
             "coincidence of the original 2 plants. If C breaks the pattern, say so plainly - do",
             "NOT cherry-pick or bury a contradicting result."]
    open(txt_path, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten: {txt_path}\nwritten: {csv_path}")


if __name__ == "__main__":
    main()
