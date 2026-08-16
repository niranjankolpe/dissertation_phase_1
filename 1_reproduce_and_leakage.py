# %% [markdown]
# # Dissertation - Day 1
# Reproduce the MLP, run the current-leakage experiment, and confirm SHAP / LIME / IG run.
#
# GOAL TODAY (not the stability harness - that comes AFTER the guide signs off on inputs):
#   1. Faithfully reproduce the 8 -> 128 -> 64 -> 1 MLP.
#   2. Produce the with/without-currents evidence table for the guide.
#   3. Smoke-test all three explainers (just confirm they run + emit an attribution vector).
#
# NOTE: uses the EXACT column names from your real CSV header.

# %% Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# %% Config -- exact column names from the real CSV
CSV_PATH = "Generation_data.csv"          # <-- adjust path if needed

TARGET = "AC Power in Watts"
ALL_FEATURES = [
    "MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)",
    "DC Current in Amps", "AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps",
]

# Three input configurations for the leakage experiment:
#   A = everything (leaky: contains AC + DC current, which are ~ power itself)
#   B = drop only the 3 AC phase currents (tests whether DC Current alone still leaks)
#   C = environment-only (no current proxies at all -> the honest estimator)
CONFIGS = {
    "A_all_8_inputs":      ALL_FEATURES,
    "B_drop_AC_currents":  ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)", "DC Current in Amps"],
    "C_environment_only":  ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)"],
}

# %% Load + clean
df = pd.read_csv(CSV_PATH)
print("Raw shape:", df.shape)
df = df.dropna().reset_index(drop=True)   # removes the trailing empty row
print("After dropna:", df.shape)
print(df.head(), "\n")
print(df.describe())

# %% Sanity checks that actually matter
# 1) Target unit: WATTS (thousands), not kW. Keep this in mind vs any 'kW' RMSE in the reference notebook.
print("Target range (W):", df[TARGET].min(), "->", df[TARGET].max())

# 2) WIND_Speed looked suspicious in the sample (very large values). Eyeball it before trusting.
print("\nWIND_Speed stats:\n", df["WIND_Speed"].describe())

# 3) Leakage check -- correlation of each feature with power.
#    Expect DC Current + the 3 AC currents to be ~0.99+ (they ARE power, via P = V*I).
corr = df[ALL_FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False)
print("\nCorrelation with", TARGET, ":\n", corr)

# %% Training helper
def train_mlp(features, data, target=TARGET, epochs=10, batch_size=32, test_size=0.30):
    X = data[features].values.astype("float32")
    y = data[target].values.astype("float32")

    # Split FIRST, scale on train only (cleaner than the reference; avoids scaler leakage).
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    model = keras.Sequential([
        layers.Input(shape=(len(features),)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64,  activation="relu"),
        layers.Dense(1,   activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    model.fit(X_tr_s, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_te_s, verbose=0).ravel()
    metrics = {
        "n_features": len(features),
        "R2":     r2_score(y_te, y_pred),
        "RMSE_W": mean_squared_error(y_te, y_pred) ** 0.5,
        "MAE_W":  mean_absolute_error(y_te, y_pred),
    }
    return model, scaler, (X_tr_s, X_te_s, y_tr, y_te), metrics

# %% THE HEADLINE: run all three configs -> this table goes to your guide
results = {}
for name, feats in CONFIGS.items():
    _, _, _, m = train_mlp(feats, df)
    results[name] = m
    print(f"{name:20s} | feats={m['n_features']} | "
          f"R2={m['R2']:.5f} | RMSE={m['RMSE_W']:.1f} W | MAE={m['MAE_W']:.1f} W")

print("\nComparison table:\n", pd.DataFrame(results).T)
# Expected story:
#   A (all 8)         -> R2 ~0.999  (leakage: current columns are power proxies)
#   B (drop AC only)  -> likely STILL very high  (DC Current alone carries the leak)
#   C (environment)   -> lower R2, honest model; the one where explanations are interesting

# %% [markdown]
# ## Explainer smoke test
# Success today = each library RUNS and emits an attribution vector over the features.
# Not real results. Provisional model = config C (the interesting one); re-point after the guide decides.

# %% Train one model to explain
SMOKE_CONFIG = "C_environment_only"
feats = CONFIGS[SMOKE_CONFIG]
model, scaler, (X_tr_s, X_te_s, y_tr, y_te), _ = train_mlp(feats, df)

N = 5
X_sample = X_te_s[:N]

def predict_fn(arr):
    return model.predict(arr, verbose=0).ravel()

# %% SHAP -- KernelExplainer (model-agnostic / perturbation-based; keeps SHAP distinct from IG)
#    (If too slow later, GradientExplainer is faster but makes SHAP gradient-based -> a methodology choice for the guide.)
import shap
background = shap.kmeans(X_tr_s, 10)
shap_expl = shap.KernelExplainer(predict_fn, background)
shap_vals = np.array(shap_expl.shap_values(X_sample, nsamples=200)).reshape(N, len(feats))
print("SHAP attributions:\n", pd.DataFrame(shap_vals, columns=feats), "\n")

# %% LIME
from lime.lime_tabular import LimeTabularExplainer
lime_expl = LimeTabularExplainer(
    X_tr_s, feature_names=feats, mode="regression", discretize_continuous=False, random_state=RANDOM_STATE
)
lime_row = np.zeros(len(feats))
exp = lime_expl.explain_instance(X_sample[0], predict_fn, num_features=len(feats))
for f_idx, w in exp.as_map()[1]:
    lime_row[f_idx] = w
print("LIME attributions (instance 0):\n", dict(zip(feats, np.round(lime_row, 4))), "\n")

# %% Integrated Gradients -- manual. BASELINE CHOICE IS AN OPEN DECISION (zeros in scaled space here).
def integrated_gradients(model, x_row, baseline=None, steps=50):
    x = tf.convert_to_tensor(x_row.reshape(1, -1), dtype=tf.float32)      # (1, F)
    if baseline is None:
        baseline = tf.zeros_like(x)                                       # (1, F) = feature means in scaled space
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1), (-1, 1))        # (S+1, 1)
    path = baseline + alphas * (x - baseline)                            # (S+1, F)
    with tf.GradientTape() as tape:
        tape.watch(path)
        preds = model(path)                                              # (S+1, 1)
    grads = tape.gradient(preds, path)                                   # (S+1, F)
    avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)   # (F,) trapezoidal
    ig = (x[0] - baseline[0]) * avg_grads                                # (F,)
    return ig.numpy()

ig_vals = np.vstack([integrated_gradients(model, X_sample[i]) for i in range(N)])
print("IG attributions:\n", pd.DataFrame(ig_vals, columns=feats))

# Completeness check: sum(IG) should ~= f(x) - f(baseline). If close, IG is implemented correctly.
f_x = float(predict_fn(X_sample[0:1])[0])
f_base = float(predict_fn(np.zeros((1, len(feats)), dtype="float32"))[0])
print(f"\nIG completeness check (instance 0): sum(IG)={ig_vals[0].sum():.2f}  vs  f(x)-f(base)={f_x - f_base:.2f}")
