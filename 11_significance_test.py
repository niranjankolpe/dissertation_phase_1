#!/usr/bin/env python3
"""Statistical test for Pillar 1 (context_transfer.txt Section 15): is Group A
(correlated AC currents) really less rank-stable than Group B (control weather
features) across seeds, or could the gap be noise? Flagged as a next step
after the 5-seed subgroup_instability result.

Uses the raw per-seed-pair rows already written by 6_subgroup_instability.py
(no retraining, no new data collection). Two tests, both nonparametric
(small n, no normality assumption):
  1. Mann-Whitney U on the pooled spearman values (A's 10 seed-pairs x 5
     methods = 50 values, vs B's 50 values) - is A stochastically lower?
  2. A permutation test on the mean(A) - mean(B) gap specifically, since
     Mann-Whitney treats all 50 as independent when they are not fully
     independent (same 5 methods, same 10 seed-pairs) - the permutation
     test instead shuffles the group label 10000 times to build a null
     distribution for the observed gap size.

Run: python 11_significance_test.py --csv analysis_notes/<ts>_subgroup_instability.csv
Output: analysis_notes/<ts>_significance_test.txt"""
import argparse, datetime
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu


def permutation_test(a, b, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    observed = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = np.mean(pooled[:na]) - np.mean(pooled[na:])
        if diff <= observed:
            count += 1
    return observed, count / n_perm


def main(a):
    df = pd.read_csv(a.csv)
    A = df[df.group == "A_correlated_leaky"]["spearman"].dropna().values
    B = df[df.group == "B_control_leak_aware"]["spearman"].dropna().values

    u_stat, u_p = mannwhitneyu(A, B, alternative="less")
    obs_diff, perm_p = permutation_test(A, B, n_perm=a.n_perm, seed=0)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    lines = [
        f"[{ts}] Significance test: Group A (correlated) vs Group B (control) spearman",
        f"Source: {a.csv}",
        f"n(A)={len(A)}  mean(A)={A.mean():.4f}  n(B)={len(B)}  mean(B)={B.mean():.4f}",
        "",
        f"Mann-Whitney U (H1: A stochastically LESS than B): U={u_stat:.1f}  p={u_p:.5f}",
        f"Permutation test (H1: mean(A)-mean(B) <= observed, {a.n_perm} shuffles): "
        f"observed diff={obs_diff:.4f}  p={perm_p:.5f}",
        "",
        "CAVEAT: both tests treat the 50 (5 methods x 10 seed-pairs) values per group as",
        "exchangeable observations, which they are NOT strictly independent (same 5 methods",
        "reused across all 10 seed-pairs, same 10 seed-pairs reused across all 5 methods).",
        "This inflates the effective sample size and can make p-values look better than they",
        "really are. Treat this as supporting/directional evidence, NOT a rigorous independent-",
        "sample test. A cleaner (but much more expensive) test would need independently trained",
        "seed pairs per comparison. State this limitation explicitly if quoting a p-value.",
    ]
    out_path = f"analysis_notes/{ts}_significance_test.txt"
    open(out_path, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="analysis_notes/2026-09-23_203425_subgroup_instability.csv")
    p.add_argument("--n_perm", type=int, default=10000)
    main(p.parse_args())
