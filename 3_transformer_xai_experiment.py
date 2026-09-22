#!/usr/bin/env python3
"""Transformer on PV data, leaky vs leak-aware, explained by SHAP/LIME/IG/ICE/PDP.
Run: python run_experiment.py --csv Generation_data.csv   (add --quick to smoke test)
Output: results/REPORT.txt + csv files."""
import argparse, itertools, math, os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
import torch, torch.nn as nn
warnings.filterwarnings("ignore")

TARGET = "AC Power in Watts"
WEATHER = ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)"]
FEATSETS = {"leaky": WEATHER + ["DC Current in Amps", "AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps"],
            "leak_aware": WEATHER}
LOCAL = ["SHAP", "LIME", "IG", "ICE"]
ALL = LOCAL + ["PDP"]


def assign_days(irr, thr=12.0):
    """Nights (low irradiance) separate days; data is chronological."""
    night = pd.Series(irr).rolling(3, center=True, min_periods=1).mean().values < thr
    day, d, was_night = np.zeros(len(irr), int), 0, True
    for i, n in enumerate(night):
        if n: was_night = True
        elif was_night: d += 1; was_night = False
        day[i] = d
    return day


def windows(X, y, day, W):
    """Windows of W timesteps ending at t, never crossing a night gap."""
    idx = np.array([t for t in range(W - 1, len(X)) if day[t] > 0 and day[t - W + 1] == day[t]])
    return np.stack([X[t - W + 1:t + 1] for t in idx]).astype(np.float32), y[idx].astype(np.float32), day[idx]


def split_days(days, f=(.7, .1)):
    u = np.unique(days); n = len(u)
    a = max(1, min(int(f[0] * n), n - 2)); b = min(n - 1, max(a + 1, int((f[0] + f[1]) * n)))
    m = lambda S: np.isin(days, S)
    return m(u[:a]), m(u[a:b]), m(u[b:])


class Net(nn.Module):
    def __init__(s, F, W, d=64):
        super().__init__()
        s.inp = nn.Linear(F, d); s.pos = nn.Parameter(torch.zeros(1, W, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True), 2)
        s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(s, x):
        return s.head(s.enc(s.inp(x) + s.pos)[:, -1]).squeeze(-1)


def train(Xtr, ytr, Xva, yva, seed, epochs, patience):
    torch.manual_seed(seed)
    m = Net(Xtr.shape[2], Xtr.shape[1]); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    Xtr, ytr, Xva, yva = map(torch.tensor, (Xtr, ytr, Xva, yva))
    best, state, bad = 1e18, None, 0
    for ep in range(epochs):
        m.train()
        for b in torch.randperm(len(Xtr)).split(256):
            opt.zero_grad(); nn.functional.mse_loss(m(Xtr[b]), ytr[b]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad(): vl = nn.functional.mse_loss(m(Xva), yva).item()
        print(f"    epoch {ep+1} val_mse={vl:.4f}", flush=True)
        if vl < best - 1e-5: best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); m.eval(); return m


def predictor(m):
    def f(X):
        with torch.no_grad():
            return np.concatenate([m(torch.tensor(X[i:i + 4096], dtype=torch.float32)).numpy()
                                   for i in range(0, len(X), 4096)])
    return f


def shap_vals(f, X, bg):
    """Exact Shapley over feature groups (feature masked across whole window)."""
    N, W, F = X.shape
    C = np.array(list(itertools.product([0, 1], repeat=F)), bool)
    v = np.zeros((N, len(C)))
    for b in bg:
        x = np.repeat(X[:, None], len(C), 1)
        mix = np.where(C[None, :, None, :], x, b).reshape(-1, W, F)
        v += f(mix).reshape(N, len(C))
    v /= len(bg)
    key = {tuple(c): i for i, c in enumerate(C)}; fac = [math.factorial(i) for i in range(F + 1)]
    phi = np.zeros((N, F))
    for j in range(F):
        for c in C:
            if c[j]: continue
            s = c.sum(); c1 = c.copy(); c1[j] = True
            phi[:, j] += fac[s] * fac[F - s - 1] / fac[F] * (v[:, key[tuple(c1)]] - v[:, key[tuple(c)]])
    return phi


def lime_vals(f, X, base, n, seed):
    """LIME over binary feature masks, weighted ridge."""
    rng = np.random.default_rng(seed); out = np.zeros((len(X), X.shape[2]))
    for i, x in enumerate(X):
        Z = rng.integers(0, 2, (n, X.shape[2])).astype(bool); Z[0] = True
        y = f(np.where(Z[:, None, :], x[None], base[None]).astype(np.float32))
        w = np.exp(-((1 - Z.sum(1) / X.shape[2]) ** 2) / .0625)
        out[i] = Ridge(1e-3).fit(Z.astype(float), y, sample_weight=w).coef_
    return out


def ig_vals(m, X, base, steps=50):
    """Integrated Gradients; returns per-feature and per-lag attribution."""
    F, W = X.shape[2], X.shape[1]; feat = np.zeros((len(X), F)); lag = np.zeros((len(X), W))
    b = torch.tensor(base, dtype=torch.float32)
    for i in range(0, len(X), 256):
        x = torch.tensor(X[i:i + 256], dtype=torch.float32); tot = torch.zeros_like(x)
        for a in torch.linspace(0, 1, steps + 1)[1:]:
            p = (b + a * (x - b)).requires_grad_(True)
            tot += torch.autograd.grad(m(p).sum(), p)[0]
        at = ((x - b) * tot / steps).numpy(); feat[i:i + 256] = at.sum(1); lag[i:i + 256] = at.sum(2)
    return feat, lag


def ice_pdp(f, X, grid):
    """Offset each feature across the window; ICE = per-instance range, PDP = range of mean."""
    N, W, F = X.shape; cur = np.zeros((F, N, len(grid)))
    for j in range(F):
        b = np.repeat(X[:, None], len(grid), 1).copy(); b[..., j] += grid[None, :, None]
        cur[j] = f(b.reshape(-1, W, F)).reshape(N, len(grid))
    pm = cur.mean(1)
    return (cur.max(2) - cur.min(2)).T, pm.max(1) - pm.min(1)


def agree(a, b, k=2):
    a, b = np.abs(a), np.abs(b)
    r = spearmanr(a, b).correlation
    return (0. if r is None or np.isnan(r) else float(r),
            float(np.argmax(a) == np.argmax(b)),
            len(set(np.argsort(a)[-k:]) & set(np.argsort(b)[-k:])) / k)


def stability(fn, X, eps, reps, seed):
    """Mean attribution change per unit input change (lower = steadier)."""
    rng = np.random.default_rng(seed); u = lambda v: v / (np.linalg.norm(v) + 1e-12)
    base = fn(X); out = []
    for _ in range(reps):
        nz = eps * rng.standard_normal(X.shape).astype(np.float32); p = fn(X + nz)
        out += [np.linalg.norm(u(p[i]) - u(base[i])) / (np.linalg.norm(nz[i]) + 1e-12) for i in range(len(X))]
    return float(np.mean(out))


def main(a):
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv).dropna().reset_index(drop=True)
    df["WIND_Speed"] /= 100.0  # raw values are cm/s (confirmed by Uzel 2026)
    day = assign_days(df["IRR (W/m2)"].values)
    acc, imp, ga, la, st, lg = [], [], [], [], [], []

    for seed in a.seeds:
        for fs, feats in FEATSETS.items():
            print(f"[seed {seed}] {fs}", flush=True)
            Xw, yw, dw = windows(df[feats].values.astype(np.float32), df[TARGET].values, day, a.window)
            tr, va, te = split_days(dw)
            sx = StandardScaler().fit(Xw[tr].reshape(-1, len(feats)))
            sc = lambda A: sx.transform(A.reshape(-1, len(feats))).reshape(A.shape).astype(np.float32)
            Xtr, Xva, Xte = sc(Xw[tr]), sc(Xw[va]), sc(Xw[te])
            ym, ys = yw[tr].mean(), yw[tr].std()

            score = lambda name, pred: (acc.append({"seed": seed, "featureset": fs, "model": name,
                "R2": r2_score(yw[te], pred), "RMSE_W": np.sqrt(mean_squared_error(yw[te], pred)),
                "MAE_W": mean_absolute_error(yw[te], pred)}), print(f"  {name} R2={acc[-1]['R2']:.4f}", flush=True))

            # tree baseline on the current row only (comparable to Uzel 2026)
            hgb = HistGradientBoostingRegressor(random_state=seed).fit(Xtr[:, -1, :], (yw[tr] - ym) / ys)
            score("HistGBDT", hgb.predict(Xte[:, -1, :]) * ys + ym)

            m = train(Xtr, (yw[tr] - ym) / ys, Xva, (yw[va] - ym) / ys, seed, a.epochs, a.patience)
            f = predictor(m)
            score("Transformer", f(Xte) * ys + ym)

            rng = np.random.default_rng(seed)
            pool = np.where(Xw[te][:, -1, feats.index("IRR (W/m2)")] > 200)[0]
            Xe = Xte[rng.choice(pool, min(a.n_explain, len(pool)), replace=False)]
            bg = Xtr[rng.choice(len(Xtr), a.n_bg, replace=False)]
            base = Xtr.mean(0).astype(np.float32); grid = np.linspace(-2, 2, 15)

            loc = {"SHAP": shap_vals(f, Xe, bg), "LIME": lime_vals(f, Xe, base, a.lime_n, seed)}
            loc["IG"], iglag = ig_vals(m, Xe, base, a.ig_steps)
            loc["ICE"], pdp = ice_pdp(f, Xe, grid)
            glob = {k: np.abs(v).mean(0) for k, v in loc.items()}; glob["PDP"] = pdp

            for k, v in glob.items():
                imp.append({"seed": seed, "featureset": fs, "method": k, "top_feature": feats[int(np.argmax(v))],
                            "concentration": float(np.abs(v).max() / np.abs(v).sum()),
                            **{ft: float(x) for ft, x in zip(feats, v)}})
            for x, y in itertools.combinations(ALL, 2):
                r, t1, t2 = agree(glob[x], glob[y])
                ga.append({"seed": seed, "featureset": fs, "pair": f"{x}-{y}", "spearman": r,
                           "top1_match": t1, "top2_overlap": t2})
            for x, y in itertools.combinations(LOCAL, 2):
                r = np.array([agree(loc[x][i], loc[y][i]) for i in range(len(Xe))])
                la.append({"seed": seed, "featureset": fs, "pair": f"{x}-{y}", "spearman": r[:, 0].mean(),
                           "top1_match": r[:, 1].mean(), "top2_overlap": r[:, 2].mean()})
            for i, v in enumerate(np.abs(iglag).mean(0)):
                lg.append({"seed": seed, "featureset": fs, "lag_before_t": a.window - 1 - i, "abs_IG": float(v)})

            if a.n_stab:
                Xs = Xe[:a.n_stab]
                for k, fn in {"SHAP": lambda A: shap_vals(f, A, bg[:4]),
                              "LIME": lambda A: lime_vals(f, A, base, a.lime_n, seed + 1),
                              "IG": lambda A: ig_vals(m, A, base, a.ig_steps)[0],
                              "ICE": lambda A: ice_pdp(f, A, grid)[0]}.items():
                    st.append({"seed": seed, "featureset": fs, "method": k,
                               "instability": stability(fn, Xs, a.eps, a.stab_reps, seed)})

            for name, rows in [("accuracy", acc), ("importance", imp), ("agreement_global", ga),
                               ("agreement_local", la), ("stability", st), ("ig_per_lag", lg)]:
                pd.DataFrame(rows).to_csv(f"{a.out}/{name}.csv", index=False)

    acc, imp, ga, la = map(pd.DataFrame, (acc, imp, ga, la))
    R = ["ACCURACY (mean over seeds)", acc.groupby(["model", "featureset"])[["R2", "RMSE_W", "MAE_W"]].mean().round(4).to_string(),
         "\nTOP FEATURE per method", imp.groupby(["featureset", "method"])["top_feature"]
         .agg(lambda x: x.value_counts().to_dict()).to_string(),
         "\nCONCENTRATION (share of top feature)",
         imp.groupby(["featureset", "method"])["concentration"].mean().unstack().round(3).to_string(),
         "\nGLOBAL AGREEMENT", ga.groupby("featureset")[["spearman", "top1_match", "top2_overlap"]].mean().round(3).to_string(),
         "\nLOCAL AGREEMENT", la.groupby("featureset")[["spearman", "top1_match", "top2_overlap"]].mean().round(3).to_string()]
    if st:
        R += ["\nSTABILITY (lower = steadier)",
              pd.DataFrame(st).groupby(["featureset", "method"])["instability"].mean().unstack().round(4).to_string()]
    R.append("\nNote: leak_aware has 4 features vs 8 leaky; rank metrics on 4 items are coarse.")
    open(f"{a.out}/REPORT.txt", "w").write("\n".join(R))
    print("done ->", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="Generation_data.csv"); p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); p.add_argument("--window", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--patience", type=int, default=6)
    p.add_argument("--n_explain", type=int, default=150); p.add_argument("--n_bg", type=int, default=16)
    p.add_argument("--lime_n", type=int, default=500); p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--n_stab", type=int, default=40); p.add_argument("--stab_reps", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.05); p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.n_explain, a.n_bg, a.lime_n, a.ig_steps, a.n_stab, a.stab_reps = \
            [0], 2, 10, 4, 100, 10, 5, 2
        a.out += "_quick"
    main(a)
