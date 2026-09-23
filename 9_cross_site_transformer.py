#!/usr/bin/env python3
"""Real cross-site test with the actual sequence Transformer (not the row-level
HistGBDT quick check in 8_cross_site_quick.py). Trains on one plant's windows,
evaluates in-site AND zero-shot on the other plant, for leaky vs leak_aware.
Explainers NOT included here (phase 2, see context_transfer.txt) - this script
is accuracy-transfer only, to confirm the 8_cross_site_quick.py signal holds
for the actual model used in the dissertation.

Plant A = Kaggle SolarGeneration (Hassan, India, 350kWp, tropical, 5-min-ish).
Plant B = NIST Ground Array (Gaithersburg MD, USA, ~230-260kWp observed, 1-min).
Day boundaries for BOTH plants computed the same way (irradiance-based, via
3_transformer_xai_experiment.py's assign_days) for a fair apples-to-apples
windowing method, even though Plant B has real timestamps available.

Zero-shot transfer detail: when a model trained on plant X is evaluated on
plant Y, plant Y's raw features are scaled with X's StandardScaler (the scaler
the model actually learned with) and predictions are de-normalized with X's
target mean/std - i.e. the model is used exactly as if deployed as-is on a
new site, no retraining, no peeking at Y's statistics.

Run: python 9_cross_site_transformer.py [--quick] [--seed 0]
Output: results_cross_site/accuracy.csv, results_cross_site/REPORT.txt
Log: logs/9_cross_site_transformer_<ts>.log (when run via nohup)"""
import argparse, datetime, importlib.util, os
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

spec = importlib.util.spec_from_file_location("exp3", "3_transformer_xai_experiment.py")
exp3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp3)

WEATHER = ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)"]
CURRENT = ["DC Current in Amps", "AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps"]
FEATSETS = {"leak_aware": WEATHER, "leaky": WEATHER + CURRENT}
TARGET = "AC Power in Watts"


def load_A():
    df = pd.read_csv("Generation_data.csv").dropna().reset_index(drop=True)
    df["WIND_Speed"] = df["WIND_Speed"] / 100.0
    day = exp3.assign_days(df["IRR (W/m2)"].values)
    return df, day


def load_B():
    df = pd.read_csv("data/plantB_nist_ground_2017.csv").dropna().reset_index(drop=True)
    day = exp3.assign_days(df["IRR (W/m2)"].values)
    return df, day


def prep(df, day, feats, window):
    return exp3.windows(df[feats].values.astype(np.float32), df[TARGET].values, day, window)


def metrics(y_true, y_pred):
    return {"R2": r2_score(y_true, y_pred), "RMSE_W": mean_squared_error(y_true, y_pred) ** 0.5,
            "MAE_W": mean_absolute_error(y_true, y_pred)}


def main(a):
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] loading plants", flush=True)
    A, dayA = load_A()
    B, dayB = load_B()
    print(f"Plant A rows={len(A)}  Plant B rows={len(B)}", flush=True)

    rows = []
    for fs, feats in FEATSETS.items():
        print(f"\n=== featureset {fs} ===", flush=True)
        XA, yA, dA = prep(A, dayA, feats, a.window)
        XB, yB, dB = prep(B, dayB, feats, a.window)
        trA, vaA, teA = exp3.split_days(dA)
        trB, vaB, teB = exp3.split_days(dB)
        print(f"A windows: tr={trA.sum()} va={vaA.sum()} te={teA.sum()}", flush=True)
        print(f"B windows: tr={trB.sum()} va={vaB.sum()} te={teB.sum()}", flush=True)

        for site_name, X, y, d, tr, va, te in (("A", XA, yA, dA, trA, vaA, teA),
                                                 ("B", XB, yB, dB, trB, vaB, teB)):
            print(f"  training on {site_name} ({fs})...", flush=True)
            sx = exp3.StandardScaler().fit(X[tr].reshape(-1, len(feats)))
            sc = lambda M: sx.transform(M.reshape(-1, len(feats))).reshape(M.shape).astype(np.float32)
            ym, ys = y[tr].mean(), y[tr].std()
            m = exp3.train(sc(X[tr]), (y[tr] - ym) / ys, sc(X[va]), (y[va] - ym) / ys,
                            a.seed, a.epochs, a.patience)
            f = exp3.predictor(m)
            import torch
            torch.save(m.state_dict(), f"saved_models/transformer_{fs}_trainon{site_name}_seed{a.seed}.pt")

            pred_own = f(sc(X[te])) * ys + ym
            row = {"trained_on": site_name, "tested_on": site_name, "featureset": fs,
                   "seed": a.seed, "zero_shot": False, **metrics(y[te], pred_own)}
            rows.append(row); print("   ", row, flush=True)

            other_X, other_y, other_te = (XB, yB, teB) if site_name == "A" else (XA, yA, teA)
            other_name = "B" if site_name == "A" else "A"
            pred_cross = f(sc(other_X[other_te])) * ys + ym
            row = {"trained_on": site_name, "tested_on": other_name, "featureset": fs,
                   "seed": a.seed, "zero_shot": True, **metrics(other_y[other_te], pred_cross)}
            rows.append(row); print("   ", row, flush=True)

            pd.DataFrame(rows).to_csv(f"{a.out}/accuracy.csv", index=False)

    df = pd.DataFrame(rows)
    R = ["CROSS-SITE TRANSFORMER ACCURACY", df.to_string(index=False), "",
         "Pivot: R2 by (trained_on, tested_on, featureset)",
         df.pivot_table(index=["trained_on", "tested_on"], columns="featureset", values="R2").round(4).to_string()]
    open(f"{a.out}/REPORT.txt", "w").write("\n".join(R))
    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] done -> {a.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_cross_site")
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.epochs, a.patience, a.out = 2, 2, a.out + "_quick"
    main(a)
