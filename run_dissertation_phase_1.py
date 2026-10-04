"""Phase-I: leakage inflation on 4 PV plants. Run: python run_dissertation_phase_1.py -> Dissertation_Phase1_Results.xlsx
Inflation = (RMSE_safe - RMSE_leaky) / RMSE_safe ; safe = weather only, leaky = weather + same-time currents/voltages."""
import os, time, threading
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB, RandomForestRegressor as RF
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae

# --- progress: a line every 30 s with elapsed time, plant, current step and Transformer epoch ---
S, T0 = {"plant": "-", "step": "starting", "ep": ""}, time.time()
def _beat():
    while True:
        time.sleep(30); t = int(time.time() - T0)
        print(f"[{t // 60:02d}:{t % 60:02d}] {S['plant']} | {S['step']} {S['ep']}", flush=True)
threading.Thread(target=_beat, daemon=True).start()

# Smallest feature-token Transformer for tabular regression (numpy, hand-written backprop).
# Each input value -> one d-dim token; 1 encoder layer (1-head self-attention + small feed-forward, both with residual);
# mean-pool over tokens -> linear output. No layer-norm/dropout. Inputs and target are standardised internally.
class TinyTransformer:
    def __init__(self, seed=0, d=16, h=32, epochs=100, bs=512, lr=3e-3):
        self.seed, self.d, self.h, self.ep, self.bs, self.lr = seed, d, h, epochs, bs, lr

    def _init(self, F):
        r, d, h = np.random.default_rng(self.seed), self.d, self.h
        g = lambda *s: r.normal(0, 1 / np.sqrt(s[0] if len(s) == 2 else d), s).astype(np.float32)
        self.p = dict(Wt=r.normal(0, .5, (F, d)).astype(np.float32), bt=np.zeros((F, d), np.float32),
                      Wq=g(d, d), Wk=g(d, d), Wv=g(d, d), Wo=g(d, d) * .5, W1=g(d, h), b1=np.zeros(h, np.float32),
                      W2=g(h, d) * .5, b2=np.zeros(d, np.float32), wh=g(d, 1)[:, 0], bh=np.zeros((), np.float32))

    def _fwd(self, X):
        p, d = self.p, self.d
        T = X[:, :, None] * p["Wt"] + p["bt"]
        Q, K, V = T @ p["Wq"], T @ p["Wk"], T @ p["Wv"]
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(d); S = S - S.max(-1, keepdims=True)
        A = np.exp(S); A /= A.sum(-1, keepdims=True)
        O = A @ V; H = T + O @ p["Wo"]
        pre = H @ p["W1"] + p["b1"]; Z = np.maximum(pre, 0); Hf = H + Z @ p["W2"] + p["b2"]
        m = Hf.mean(1); return m @ p["wh"] + p["bh"], (X, T, Q, K, V, A, O, H, pre, Z, m)

    def _grad(self, X, t):
        p, d = self.p, self.d; y, (X, T, Q, K, V, A, O, H, pre, Z, m) = self._fwd(X); B, F = X.shape
        dy = 2 * (y - t) / B; g = {"wh": m.T @ dy, "bh": dy.sum()}
        dHf = (dy[:, None] * p["wh"])[:, None, :] / F * np.ones((B, F, d), np.float32)
        f2 = lambda a: a.reshape(-1, a.shape[-1])
        g["W2"], g["b2"] = f2(Z).T @ f2(dHf), dHf.sum((0, 1))
        dpre = (dHf @ p["W2"].T) * (pre > 0); g["W1"], g["b1"] = f2(H).T @ f2(dpre), dpre.sum((0, 1))
        dH = dHf + dpre @ p["W1"].T
        dO = dH @ p["Wo"].T; g["Wo"] = f2(O).T @ f2(dH)
        dA = dO @ V.transpose(0, 2, 1); dV = A.transpose(0, 2, 1) @ dO
        dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(d)
        dQ, dK = dS @ K, dS.transpose(0, 2, 1) @ Q
        g["Wq"], g["Wk"], g["Wv"] = f2(T).T @ f2(dQ), f2(T).T @ f2(dK), f2(T).T @ f2(dV)
        dT = dH + dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
        g["Wt"], g["bt"] = (dT * X[:, :, None]).sum(0), dT.sum(0)
        return g, float(((y - t) ** 2).mean())

    def fit(self, X, y):
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
        self.mx, self.sx, self.my, self.sy = X.mean(0), X.std(0) + 1e-6, y.mean(), y.std() + 1e-6
        X, y = (X - self.mx) / self.sx, (y - self.my) / self.sy
        self._init(X.shape[1]); r = np.random.default_rng(self.seed + 1)
        m1 = {k: np.zeros_like(v) for k, v in self.p.items()}; m2 = {k: np.zeros_like(v) for k, v in self.p.items()}; k = 0
        nb = int(np.ceil(len(X) / self.bs)); tot = self.ep * nb
        for e in range(self.ep):
            S["ep"] = f"(epoch {e + 1}/{self.ep})"; idx = r.permutation(len(X))
            for b in range(nb):
                j = idx[b * self.bs:(b + 1) * self.bs]; g, _ = self._grad(X[j], y[j]); k += 1
                lr = self.lr * .5 * (1 + np.cos(np.pi * k / tot))  # cosine decay
                for n in self.p:
                    m1[n] = .9 * m1[n] + .1 * g[n]; m2[n] = .999 * m2[n] + .001 * g[n] ** 2
                    self.p[n] -= (lr * (m1[n] / (1 - .9 ** k)) / (np.sqrt(m2[n] / (1 - .999 ** k)) + 1e-8)).astype(np.float32)
        S["ep"] = ""; return self

    def predict(self, X):
        X = (np.asarray(X, np.float32) - self.mx) / self.sx
        return np.concatenate([self._fwd(X[i:i + 4096])[0] for i in range(0, len(X), 4096)]) * self.sy + self.my

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
MODELS = {"HGB": lambda s: HGB(random_state=s),
          "RF": lambda s: RF(100, min_samples_leaf=5, n_jobs=-1, random_state=s),
          "Linear": lambda s: LR(),
          "Transformer": lambda s: TinyTransformer(s, epochs=100)}  # 1-layer feature-token Transformer, defined above

def ts(path, **kw):  # average to 5-min blocks (first column = time), chunked for the 1.4 GB file
    S, N = [], []
    for c in pd.read_csv(path, chunksize=2_000_000, **kw):
        t = pd.to_datetime(c.iloc[:, 0], utc=True).dt.floor("5min")
        c = c.iloc[:, 1:]
        S.append(c.groupby(t).sum()); N.append(c.groupby(t).count())
    return pd.concat(S).groupby(level=0).sum() / pd.concat(N).groupby(level=0).sum()

W = ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)"]
I = ["DC Current in Amps", "AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps"]
Y, IRR = "AC Power in Watts", "IRR (W/m2)"
PLANTS = {  # name: (loader, safe inputs, leaky extras, target, irradiance column)
    "A Hassan (India)": (lambda: pd.read_csv("Generation_data.csv"), W, I, Y, IRR),
    "B NIST (USA)": (lambda: ts("data/plantB_nist_ground_2017.csv"), W, I, Y, IRR),
    "C HKUST (HK)": (lambda: ts("data/plantC_hk_lsk_north.csv"), W[1:], ["AC_1 in Amps", "AC_2 in Amps", "AC_3 in Amps"], Y, IRR),
    "D La Reunion": (lambda: pd.read_csv("data/plantD_5min.csv", index_col=0, parse_dates=True).loc[:"2022-05-30 23:59:59"],
        ["GTI", "DTI", "TA", "TPV"], ["Ia", "Ig", "Va", "Vg"], "Pg", "GTI"),
}

def ev(tr, te, cols, y, m, s):  # -> [R2, RMSE, MAE], squared errors on the test block
    p = MODELS[m](s).fit(tr[cols], tr[y]).predict(te[cols])
    return [r2_score(te[y], p), mse(te[y], p) ** .5, mae(te[y], p)], (te[y].values - p) ** 2

def ci(es, el, day, B=2000):  # day-block bootstrap 95% CI of inflation (%): resample test days
    g = pd.DataFrame({"s": es, "l": el, "d": day}).groupby("d").sum(); i = np.random.default_rng(0).integers(0, len(g), (B, len(g)))
    return 100 * np.percentile(1 - np.sqrt(g.l.values[i].sum(1) / g.s.values[i].sum(1)), [2.5, 97.5])

def split(d, irr, thr):  # daytime filter then same chronological 70/30
    d = d[d[irr] > thr] if thr is not None else d
    n = int(.7 * len(d)); return d.iloc[:n].copy(), d.iloc[n:].copy()

def run(tr, te, wx, lk, y, m, day, seeds=SEEDS):  # -> dict of mean metrics + CI
    r = [(ev(tr, te, wx, y, m, s), ev(tr, te, wx + lk, y, m, s)) for s in seeds]
    a = np.array([[x[0][0], x[1][0]] for x in r]); mu = a.mean(0)
    lo, hi = ci(r[0][0][1], r[0][1][1], day)
    return dict(R2_safe=mu[0, 0], R2_leaky=mu[1, 0], RMSE_safe=mu[0, 1], RMSE_leaky=mu[1, 1], MAE_safe=mu[0, 2], MAE_leaky=mu[1, 2],
                **{"Inflation_%": 100 * (mu[0, 1] - mu[1, 1]) / mu[0, 1], "Inflation_std_%": 100 * (1 - a[:, 1, 1] / a[:, 0, 1]).std(),
                   "CI95_low_%": lo, "CI95_high_%": hi})

res, san, setup, tf, sens, roll = [], [], [], [], [], []
for pi, (name, (load, wx, lk, y, irr)) in enumerate(PLANTS.items(), 1):
    S["plant"] = f"plant {pi}/4 {name}"; S["step"] = "loading data"; print(S["plant"], flush=True)
    d = load().dropna(); stamped = isinstance(d.index, pd.DatetimeIndex)
    if stamped: d["hour"] = d.index.hour + d.index.minute / 60; d["doy"] = d.index.dayofyear
    tr, te = split(d, irr, 10)
    day = te.index.date if stamped else np.arange(len(te)) // 144          # A: 144-row pseudo-days (~12 h at 5 min)
    setup.append(dict(Plant=name, Rows=len(tr) + len(te), Train=len(tr), Test=len(te),
                      Period=f"{d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d}" if stamped else "no timestamps (row order)",
                      Safe_inputs=", ".join(wx), Leaky_extra_inputs=", ".join(lk), Test_days=len(set(day))))
    for m in MODELS:
        S["step"] = f"main results, model {m}"
        res.append(dict(Plant=name, Model=m, **run(tr, te, wx, lk, y, m, day)))
        if stamped and m != "Transformer": tf.append(dict(Plant=name, Model=m, **run(tr, te, wx + ["hour", "doy"], lk, y, m, day)))
    S["step"] = "irradiance-threshold sensitivity"
    for thr in (None, 10, 50):                                         # daytime-threshold sensitivity (HGB, seed 0)
        a, b = split(d, irr, thr); dy = b.index.date if stamped else np.arange(len(b)) // 144
        sens.append(dict(Plant=name, Irradiance_threshold=thr if thr is not None else "none", Test_rows=len(b),
                         **run(a, b, wx, lk, y, "HGB", dy, [0])))
    for k in range(1, 6):
        S["step"] = f"rolling split {k}/5"                                              # rolling splits: train on blocks 1..k, test on block k+1 of 6
        a, b = d[d[irr] > 10].iloc[:k * (len(tr) + len(te)) // 6], d[d[irr] > 10].iloc[k * (len(tr) + len(te)) // 6:(k + 1) * (len(tr) + len(te)) // 6]
        dy = b.index.date if stamped else np.arange(len(b)) // 144
        for m in ("HGB", "Linear"):
            roll.append(dict(Plant=name, Fold=k, Model=m, Train_rows=len(a), Test_rows=len(b),
                             Test_period=f"{b.index.min():%Y-%m-%d} to {b.index.max():%Y-%m-%d}" if stamped else "row order", **run(a, b, wx, lk, y, m, dy, [0])))
    S["step"] = "sanity checks"
    tr["_y"], te["_y"] = tr[y], te[y]                                  # sanity: target as feature -> ~100%; noise column -> ~0%
    rng = np.random.default_rng(0); tr["_n"], te["_n"] = rng.normal(size=len(tr)), rng.normal(size=len(te))
    rh, rl = ev(tr, te, wx, y, "HGB", 0)[0][1], ev(tr, te, wx, y, "Linear", 0)[0][1]
    san.append(dict(Plant=name, **{"Target_as_feature_%": 100 * (1 - ev(tr, te, wx + ["_y"], y, "Linear", 0)[0][1] / rl),
                                   "Noise_column_%": 100 * (1 - ev(tr, te, wx + ["_n"], y, "HGB", 0)[0][1] / rh)}))

method = pd.DataFrame({"Method": [
    "Inflation = (RMSE_safe - RMSE_leaky) / RMSE_safe. Safe = weather only; leaky = weather + same-time currents/voltages.",
    "Same chronological 70/30 split (by row/time order) for both models; metrics on the 30% test block; mean of 3 seeds.",
    "Plants B, C, D averaged to 5-min blocks; A has no timestamps (row order). Main results keep rows with irradiance > 10 W/m2.",
    "Plant D: data up to 2022-05-30 (irradiance timestamps shift after); cumulative energy (Eg) and grid frequency (Fg) dropped. Weather is measured, not forecast.",
    "CI95: day-block bootstrap (2000 resamples of test days; seed-0 errors); inflation per resample = 1 - sqrt(sum e_leaky^2 / sum e_safe^2). Plant A: 144-row blocks.",
    "TimeFeatures: hour-of-day and day-of-year added to BOTH safe and leaky inputs (B, C, D; A has no timestamps; HGB, RF, Linear only).",
    "RollingSplits: 5 expanding-window chronological splits (train on blocks 1..k of 6, test on block k+1; HGB and Linear, seed 0).",
    "Transformer: one encoder layer (1-head self-attention + small feed-forward), one token per input value, mean-pooled; hand-written numpy, 100 epochs (12 epochs left it under-trained on Plant D). Not run in TimeFeatures.",
    "Sensitivity: irradiance threshold none / 10 / 50 W/m2, HGB, seed 0.",
    "Sanity: target-as-feature should be ~100%, random noise column ~0% (target test: Linear; noise test: HGB; seed 0)."]})
with pd.ExcelWriter("Dissertation_Phase1_Results.xlsx") as w:
    for nm, df in (("Results", pd.DataFrame(res)), ("TimeFeatures", pd.DataFrame(tf)), ("Sensitivity", pd.DataFrame(sens)), ("RollingSplits", pd.DataFrame(roll)),
                   ("Sanity", pd.DataFrame(san)), ("Setup", pd.DataFrame(setup)), ("Method", method)):
        df.round(3).to_excel(w, sheet_name=nm, index=False)
c = ["Plant", "Model", "Inflation_%", "CI95_low_%", "CI95_high_%"]
print(pd.DataFrame(res)[c].round(1).to_string(), "\n", pd.DataFrame(tf)[c].round(1).to_string(), "\n",
      pd.DataFrame(sens)[["Plant", "Irradiance_threshold", "Test_rows", "R2_safe", "Inflation_%"]].round(3).to_string(), "\n", pd.DataFrame(san).round(2).to_string())
