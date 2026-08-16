# %% [markdown]
# # Dissertation - Day 2: Stability + Agreement
# Measures the two things that ARE your project, on the honest 4-input model:
#   STABILITY  = when you nudge the inputs a tiny bit, how much does each method's explanation move?
#                (lower = more stable). This is the local-Lipschitz idea (Alvarez-Melis 2018).
#   AGREEMENT  = do the three methods rank the features the same way?
#                (Spearman correlation + top-k overlap).
#
# Tuned SMALL so it finishes on a CPU laptop. Raise the knobs later once it works.
# CPU note: SHAP + LIME are the slow part (they re-run the model many times, inside the
# stability loop). Keep N_INSTANCES and N_PERTURB small until you know it runs.

# %% Knobs (these are your open decisions - tune later, justify in the report)
N_INSTANCES   = 15      # how many test rows to test on
N_PERTURB     = 5       # how many tiny nudges per row (for stability)
EPSILON       = 0.05    # nudge size in SCALED space (inputs have std=1, so this is a small nudge)
TOP_K         = 2       # top-k overlap for agreement
SHAP_NSAMPLES = 100     # KernelSHAP samples (higher = more accurate, slower)
IG_STEPS      = 50      # integration steps for IG
RANDOM_STATE  = 42

# %% Imports
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)

CSV_PATH = "Generation_data.csv"
TARGET   = "AC Power in Watts"
FEATURES = ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)"]   # the honest 4-input model
F = len(FEATURES)

# %% Build the honest model
df = pd.read_csv(CSV_PATH).dropna().reset_index(drop=True)
X = df[FEATURES].values.astype("float32"); y = df[TARGET].values.astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

model = keras.Sequential([
    layers.Input(shape=(F,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64,  activation="relu"),
    layers.Dense(1,   activation="linear"),
])
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
model.fit(Xtr_s, ytr, epochs=10, batch_size=32, verbose=0)
print("Honest model R2:", round(r2_score(yte, model.predict(Xte_s, verbose=0).ravel()), 4))

def predict_fn(arr):
    return model.predict(np.asarray(arr, dtype="float32"), verbose=0).ravel()

# %% The three explainers -- each returns one attribution vector (length F) for a scaled row
import shap
from lime.lime_tabular import LimeTabularExplainer

_bg = shap.kmeans(Xtr_s, 10)
_shap = shap.KernelExplainer(predict_fn, _bg)
def attr_shap(x):
    v = _shap.shap_values(x.reshape(1, -1), nsamples=SHAP_NSAMPLES, silent=True)
    return np.array(v).reshape(F)

_lime = LimeTabularExplainer(Xtr_s, feature_names=FEATURES, mode="regression",
                             discretize_continuous=False, random_state=RANDOM_STATE)
def attr_lime(x):
    e = _lime.explain_instance(x, predict_fn, num_features=F)
    out = np.zeros(F)
    for idx, w in e.as_map()[1]:
        out[idx] = w
    return out

def attr_ig(x, baseline=None):
    xt = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(xt)            # IG BASELINE = zeros in scaled space (an open choice)
    a = tf.reshape(tf.linspace(0.0, 1.0, IG_STEPS + 1), (-1, 1))
    path = baseline + a * (xt - baseline)
    with tf.GradientTape() as tape:
        tape.watch(path); preds = model(path)
    g = tape.gradient(preds, path)
    avg = tf.reduce_mean((g[:-1] + g[1:]) / 2.0, axis=0)
    return ((xt[0] - baseline[0]) * avg).numpy()

METHODS = {"SHAP": attr_shap, "LIME": attr_lime, "IG": attr_ig}

# %% Pick the test instances
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(Xte_s), size=N_INSTANCES, replace=False)
instances = Xte_s[idx]

# %% STABILITY: how much does the explanation move when inputs are nudged a little?
# score = ||attr(x) - attr(x')|| / ||x - x'||   (mean over small nudges, then over rows)
#
# We report TWO versions:
#   RAW        - uses each method's attributions as-is.
#   NORMALISED - unit-scales each attribution vector first, so the comparison reflects a
#                CHANGE IN SHAPE, not a difference in magnitude between methods. This is the
#                FAIR cross-method number (SHAP/IG produce big values, LIME small ones, so raw
#                magnitudes aren't directly comparable). Rank the methods by NORMALISED.
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

print("\n=== STABILITY (lower = more stable) ===")
stab_raw, stab_norm = {}, {}
for name, fn in METHODS.items():
    r_raw, r_norm = [], []
    for x in instances:
        base = fn(x)
        base_u = _unit(base)
        for _ in range(N_PERTURB):
            xp = x + EPSILON * rng.standard_normal(F).astype("float32")
            den = np.linalg.norm(xp - x) + 1e-12
            ap = fn(xp)
            r_raw.append(np.linalg.norm(ap - base) / den)
            r_norm.append(np.linalg.norm(_unit(ap) - base_u) / den)
    stab_raw[name]  = float(np.mean(r_raw))
    stab_norm[name] = float(np.mean(r_norm))
    print(f"  {name:5s}:  raw={stab_raw[name]:10.4f}   normalised={stab_norm[name]:.4f}")

rank_norm = sorted(stab_norm, key=stab_norm.get)
print("  Ranking by NORMALISED (fair, most stable first):", " < ".join(rank_norm))

# %% AGREEMENT: do the methods rank features the same way? (per row, then averaged)
print("\n=== AGREEMENT (higher = methods agree) ===")
pairs = [("SHAP", "LIME"), ("SHAP", "IG"), ("LIME", "IG")]
spear = {p: [] for p in pairs}
topk  = {p: [] for p in pairs}
for x in instances:
    imp = {m: np.abs(fn(x)) for m, fn in METHODS.items()}   # importance = |attribution|
    for a, b in pairs:
        rho = spearmanr(imp[a], imp[b]).correlation
        spear[(a, b)].append(0.0 if np.isnan(rho) else rho)
        ta = set(np.argsort(imp[a])[-TOP_K:]); tb = set(np.argsort(imp[b])[-TOP_K:])
        topk[(a, b)].append(len(ta & tb) / TOP_K)
for a, b in pairs:
    print(f"  {a:4s} vs {b:4s}:  Spearman={np.mean(spear[(a,b)]):.3f}   top-{TOP_K} overlap={np.mean(topk[(a,b)]):.2f}")

print("\nDone. First real results: which method is steadiest, and which two agree most.")
