#!/usr/bin/env python3
"""Extend results/ with additional seeds WITHOUT redoing existing seeds 0,1,2.
Reuses training/explainer code from 3_transformer_xai_experiment.py unchanged.
Run: python 5_extra_seeds.py --seeds 3 4
Writes into results_new_seeds/ first, then merges into results/*.csv and
rewrites results/REPORT.txt from the combined (old+new) data.
Logs progress with timestamps to logs/5_extra_seeds_<ts>.log"""
import argparse, importlib.util, datetime, os
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("exp3", "3_transformer_xai_experiment.py")
exp3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp3)

FILES = ["accuracy", "importance", "agreement_global", "agreement_local", "stability", "ig_per_lag"]


def merge_and_report(old_dir, new_dir, out_dir):
    frames = {}
    for name in FILES:
        parts = []
        for d in (old_dir, new_dir):
            p = f"{d}/{name}.csv"
            if os.path.exists(p):
                parts.append(pd.read_csv(p))
        merged = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()
        merged.to_csv(f"{out_dir}/{name}.csv", index=False)
        frames[name] = merged

    acc, imp, ga, la, st = (frames["accuracy"], frames["importance"], frames["agreement_global"],
                             frames["agreement_local"], frames["stability"])
    R = ["ACCURACY (mean over seeds)", acc.groupby(["model", "featureset"])[["R2", "RMSE_W", "MAE_W"]].mean().round(4).to_string(),
         "\nTOP FEATURE per method", imp.groupby(["featureset", "method"])["top_feature"]
         .agg(lambda x: x.value_counts().to_dict()).to_string(),
         "\nCONCENTRATION (share of top feature)",
         imp.groupby(["featureset", "method"])["concentration"].mean().unstack().round(3).to_string(),
         "\nGLOBAL AGREEMENT", ga.groupby("featureset")[["spearman", "top1_match", "top2_overlap"]].mean().round(3).to_string(),
         "\nLOCAL AGREEMENT", la.groupby("featureset")[["spearman", "top1_match", "top2_overlap"]].mean().round(3).to_string()]
    if len(st):
        R += ["\nSTABILITY (lower = steadier)",
              st.groupby(["featureset", "method"])["instability"].mean().unstack().round(4).to_string()]
    R.append(f"\nSeeds included: {sorted(acc['seed'].unique().tolist())}")
    R.append("\nNote: leak_aware has 4 features vs 8 leaky; rank metrics on 4 items are coarse.")
    open(f"{out_dir}/REPORT.txt", "w").write("\n".join(R))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="Generation_data.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--patience", type=int, default=6)
    p.add_argument("--n_explain", type=int, default=150); p.add_argument("--n_bg", type=int, default=16)
    p.add_argument("--lime_n", type=int, default=500); p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--n_stab", type=int, default=40); p.add_argument("--stab_reps", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.05)
    a = p.parse_args()
    a.quick = False
    a.out = "results_new_seeds"
    os.makedirs("logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] starting seeds {a.seeds} -> {a.out}", flush=True)
    exp3.main(a)
    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] merging {a.out} into results/", flush=True)
    merge_and_report("results", a.out, "results")
    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] done. results/REPORT.txt updated with seeds "
          f"{a.seeds} added.", flush=True)
