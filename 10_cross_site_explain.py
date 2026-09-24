#!/usr/bin/env python3
"""Phase 2 (flagged as next step in context_transfer.txt Section 16/9-D/G):
does the EXPLANATION transfer across sites, not just accuracy?

Loads the already-trained models from saved_models/ (script 9, seed 0) - no
retraining. For each trained model, runs the same 5 explainers (SHAP/LIME/
IG/ICE/PDP, reusing 3_transformer_xai_experiment.py's exact implementations)
on its OWN home-site test data AND on the OTHER site's test data (always
scaled with the training site's scaler - i.e. explaining exactly what the
deployed model sees, consistent with 9_cross_site_transformer.py's zero-shot
setup). Then measures: does the explanation (top feature, concentration,
ranking) change when the same frozen model explains a foreign site's data
vs its own?

Run: python 10_cross_site_explain.py
Output: results_cross_site_explain/cross_site_importance.csv
        results_cross_site_explain/cross_site_explanation_agreement.csv
        results_cross_site_explain/REPORT.txt"""
import argparse, datetime, importlib.util, os
import numpy as np, pandas as pd
import torch

spec = importlib.util.spec_from_file_location("exp3", "3_transformer_xai_experiment.py")
exp3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp3)

WEATHER = ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)"]
CURRENT = ["DC Current in Amps", "AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps"]
FEATSETS = {"leak_aware": WEATHER, "leaky": WEATHER + CURRENT}
TARGET = "AC Power in Watts"
ALL = ["SHAP", "LIME", "IG", "ICE", "PDP"]


def load_A():
    df = pd.read_csv("Generation_data.csv").dropna().reset_index(drop=True)
    df["WIND_Speed"] = df["WIND_Speed"] / 100.0
    return df, exp3.assign_days(df["IRR (W/m2)"].values)


def load_B():
    df = pd.read_csv("data/plantB_nist_ground_2017.csv").dropna().reset_index(drop=True)
    return df, exp3.assign_days(df["IRR (W/m2)"].values)


def prep(df, day, feats, window):
    return exp3.windows(df[feats].values.astype(np.float32), df[TARGET].values, day, window)


def main(a):
    os.makedirs(a.out, exist_ok=True)
    plants_raw = {"A": load_A(), "B": load_B()}
    rows_imp, rows_agree = [], []
    print(f"[{datetime.datetime.now().strftime('%H%M%S')}] start", flush=True)

    for fs, feats in FEATSETS.items():
        F = len(feats)
        prepped, scalers = {}, {}
        for site, (df, day) in plants_raw.items():
            X, y, d = prep(df, day, feats, a.window)
            tr, va, te = exp3.split_days(d)
            sx = exp3.StandardScaler().fit(X[tr].reshape(-1, F))
            prepped[site] = (X, y, d, tr, va, te)
            scalers[site] = sx
        print(f"  [{fs}] windows prepped", flush=True)

        for train_site in ("A", "B"):
            print(f"  [{fs}] loading model trained on {train_site}", flush=True)
            m = exp3.Net(F, a.window)
            m.load_state_dict(torch.load(f"saved_models/transformer_{fs}_trainon{train_site}_seed0.pt"))
            m.eval()
            f = exp3.predictor(m)
            sx = scalers[train_site]
            sc = lambda M: sx.transform(M.reshape(-1, F)).reshape(M.shape).astype(np.float32)

            Xh, yh, dh, trh, vah, teh = prepped[train_site]
            rng = np.random.default_rng(0)
            bg_pool = Xh[trh]
            bg = sc(bg_pool[rng.choice(len(bg_pool), a.n_bg, replace=False)])
            base = sc(bg_pool).mean(0).astype(np.float32)
            grid = np.linspace(-2, 2, 15)

            globs = {}
            for explain_site in ("A", "B"):
                X, y, d, tr, va, te = prepped[explain_site]
                irr_idx = feats.index("IRR (W/m2)")
                pool = np.where(X[te][:, -1, irr_idx] > 200)[0]
                idx = rng.choice(pool, min(a.n_explain, len(pool)), replace=False)
                Xe = sc(X[te][idx])

                print(f"    [{fs}] trainon={train_site} explain={explain_site} "
                      f"({'HOME' if explain_site==train_site else 'FOREIGN'}) n={len(Xe)}", flush=True)
                loc = {"SHAP": exp3.shap_vals(f, Xe, bg), "LIME": exp3.lime_vals(f, Xe, base, a.lime_n, 0)}
                loc["IG"], _ = exp3.ig_vals(m, Xe, base, a.ig_steps)
                loc["ICE"], pdp = exp3.ice_pdp(f, Xe, grid)
                glob = {k: np.abs(v).mean(0) for k, v in loc.items()}
                glob["PDP"] = pdp
                globs[explain_site] = glob

                for k, v in glob.items():
                    rows_imp.append({"featureset": fs, "train_site": train_site, "explain_site": explain_site,
                                      "home_or_foreign": "home" if explain_site == train_site else "foreign",
                                      "method": k, "top_feature": feats[int(np.argmax(v))],
                                      "concentration": float(np.abs(v).max() / np.abs(v).sum())})

            for method in ALL:
                r, t1, t2 = exp3.agree(globs["A"][method], globs["B"][method], k=2)
                rows_agree.append({"featureset": fs, "train_site": train_site, "method": method,
                                    "spearman_A_vs_B_explanation": r, "top1_match": t1, "top2_overlap": t2})

            pd.DataFrame(rows_imp).to_csv(f"{a.out}/cross_site_importance.csv", index=False)
            pd.DataFrame(rows_agree).to_csv(f"{a.out}/cross_site_explanation_agreement.csv", index=False)

    imp = pd.DataFrame(rows_imp)
    agr = pd.DataFrame(rows_agree)
    R = ["CROSS-SITE EXPLANATION TRANSFER",
         "\nTop feature by (featureset, train_site, explain_site, method):",
         imp.pivot_table(index=["featureset", "train_site", "method"], columns="explain_site",
                          values="top_feature", aggfunc="first").to_string(),
         "\nConcentration by (featureset, train_site, home/foreign, method):",
         imp.pivot_table(index=["featureset", "train_site", "method"], columns="home_or_foreign",
                          values="concentration").round(3).to_string(),
         "\nAgreement between explaining home data vs foreign data (SAME frozen model):",
         agr.to_string(index=False),
         "\nREAD: spearman close to 1 = explanation ranking barely changes when the model is",
         "pointed at a foreign site's data. Low spearman / top1 mismatch = the SAME model",
         "explains itself differently depending on which site's data it is shown - i.e.",
         "explanations do NOT transfer as cleanly as accuracy did."]
    open(f"{a.out}/REPORT.txt", "w").write("\n".join(R))
    print(f"[{datetime.datetime.now().strftime('%H%M%S')}] done -> {a.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_cross_site_explain")
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--n_explain", type=int, default=150)
    p.add_argument("--n_bg", type=int, default=16)
    p.add_argument("--lime_n", type=int, default=500)
    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.n_explain, a.n_bg, a.lime_n, a.ig_steps, a.out = 10, 4, 100, 10, a.out + "_quick"
    main(a)
