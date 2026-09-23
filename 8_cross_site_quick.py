#!/usr/bin/env python3
"""Cheap feasibility gate for the cross-site extension (context_transfer.txt
Section 9): does leakage help/hurt when a model trained on one plant is
tested on a totally different plant? Row-level HistGBDT only (no Transformer,
no windowing) - this is a fast go/no-go check, not the final experiment.

Plant A = Kaggle SolarGeneration, Hassan, Karnataka, India, 350kWp (tropical).
Plant B = NIST Campus Ground Array, Gaithersburg MD, USA, ~230-260kWp
          observed peak (temperate, four seasons, snow in winter).
Two very different climates/capacities/hemispheres/units - a real
generalization test, not a rerun of the same distribution.

Run: python 8_cross_site_quick.py
Output: analysis_notes/<ts>_cross_site_quick.txt"""
import datetime, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

WEATHER = ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)"]
CURRENT = ["DC Current in Amps", "AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps"]
FEATSETS = {"leak_aware": WEATHER, "leaky": WEATHER + CURRENT}
TARGET = "AC Power in Watts"
SEED = 42


def load_A():
    df = pd.read_csv("Generation_data.csv").dropna().reset_index(drop=True)
    df["WIND_Speed"] = df["WIND_Speed"] / 100.0
    return df


def load_B():
    return pd.read_csv("data/plantB_nist_ground_2017.csv").dropna().reset_index(drop=True)


def fit_eval(train_df, test_df, feats, same_site, cap_ref):
    Xtr, ytr = train_df[feats].values, train_df[TARGET].values
    if same_site:
        Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, test_size=0.3, random_state=SEED)
    else:
        Xte, yte = test_df[feats].values, test_df[TARGET].values
    m = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr, ytr)
    pred = m.predict(Xte)
    r2 = r2_score(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    return r2, rmse, rmse / cap_ref


def main():
    A, B = load_A(), load_B()
    cap_A = np.percentile(A[TARGET], 99.5)
    cap_B = np.percentile(B[TARGET], 99.5)
    rows = []
    combos = [("A_in_site", A, A, True, cap_A), ("B_in_site", B, B, True, cap_B),
              ("A_train_B_test (zero-shot)", A, B, False, cap_B),
              ("B_train_A_test (zero-shot)", B, A, False, cap_A)]
    for name, tr, te, same, cap in combos:
        for fs, feats in FEATSETS.items():
            r2, rmse, nrmse = fit_eval(tr, te, feats, same, cap)
            rows.append({"combo": name, "featureset": fs, "R2": r2, "RMSE_W": rmse,
                         "RMSE_over_p99.5cap": nrmse})
            print(f"{name:30s} {fs:11s} R2={r2:8.4f}  RMSE={rmse:10.1f} W  "
                  f"RMSE/cap={nrmse:.4f}", flush=True)

    out = pd.DataFrame(rows)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs("analysis_notes", exist_ok=True)
    txt_path = f"analysis_notes/{ts}_cross_site_quick.txt"
    csv_path = f"analysis_notes/{ts}_cross_site_quick.csv"
    out.to_csv(csv_path, index=False)
    lines = [f"[{ts}] Cross-site feasibility check (HistGBDT, row-level, no windowing)",
             f"Plant A rows: {len(A)}  (p99.5 power = {cap_A:.0f} W)",
             f"Plant B rows: {len(B)}  (p99.5 power = {cap_B:.0f} W)",
             "", out.to_string(index=False),
             "",
             "READ: R2 can go very negative on zero-shot cross-site (different capacity/",
             "climate/units) - that is expected and not itself a failure of the idea.",
             "What matters for the Section 9 extension: does the leaky/leak_aware GAP",
             "(in R2 or RMSE/cap) behave differently in-site vs cross-site? If leakage's",
             "advantage shrinks/reverses cross-site, that is the finding worth pursuing",
             "with the full Transformer + windowing + 5-explainer pipeline next."]
    open(txt_path, "w").write("\n".join(lines))
    print(f"\nwritten: {txt_path}\nwritten: {csv_path}")


if __name__ == "__main__":
    main()
