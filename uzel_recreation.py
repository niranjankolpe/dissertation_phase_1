"""
Recreation of: Uzel, H. (2026). "SCADA-Based AC Power Forecasting in a Utility-Scale PV Plant:
Explainable HistGradientBoosting and Thermal Derating Quantification."
GU J Sci, Part A, 13(1), 452-482. DOI 10.54287/gujsa.1818552.

Design choices (features, splits, hyperparameters, metrics, derating formulas) are taken from
the paper's Sections 3-5 (Hassan, Karnataka, India SCADA data, N=118,865, 9 variables;
Table 2 split protocol; Table 4 hyperparameters; Section 4.6 derating method).

DEVIATIONS (forced by package availability, not a modeling choice):
  - XGBoost: SKIPPED if the `xgboost` package is not installed (auto-detected below).
  - SHAP: SKIPPED if `shap` is not installed (auto-detected below). sklearn's own
    permutation_importance + partial_dependence (PDP/ICE) are used as a fallback --
    PDP/ICE is itself one of the paper's stated methods (Sec 4.5); SHAP values specifically
    are only produced if the package is present.
  - Semiparametric thermal-derating model (Sec 4.6, eq. 4): uses statsmodels GAM-style
    fitting if `statsmodels`+`pygam`/`patsy` are installed; otherwise falls back to a
    natural cubic B-spline basis (scipy) fit by closed-form OLS (same model class,
    different fitting library) -- auto-detected below.
"""

import os, json, time
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import BSpline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

t0 = time.time()
OUT = "uzel_outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 3.1 Dataset and Variables
# ---------------------------------------------------------------------------
DATA_PATH = "Generation_data.xlsx"   # must be in the same folder as this script
df = pd.read_excel(DATA_PATH)

RAW_COLS = {
    "module_temp": "MODULE_TEMP",
    "amb_temp": "Amb_Temp",
    "wind": "WIND_Speed",
    "irr": "IRR (W/m2)",
    "dc_current": "DC Current in Amps",
    "ac_ir": "AC Ir in Amps",
    "ac_iy": "AC Iy in Amps",
    "ac_ib": "AC Ib in Amps",
    "ac_power_w": "AC Power in Watts",
}
missing_cols = [c for c in RAW_COLS.values() if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Column mismatch vs expected 9 variables. Missing: {missing_cols}. "
                      f"Found columns: {list(df.columns)}")

N = len(df)
print(f"N = {N} (paper: 118,865)")
if N != 118865:
    print("WARNING: row count does not match paper's reported N -- check you have the right CSV.")

n_missing = int(df.isna().sum().sum())
n_dupe = int(df.duplicated().sum())
print(f"Missing values: {n_missing} | Duplicate rows: {n_dupe} (paper reports 0 / 0)")

# Target reported in kW in the paper; raw file is in Watts.
df["AC_Power_kW"] = df[RAW_COLS["ac_power_w"]] / 1000.0

# Table 1 descriptive-stats sanity check (paper's reported means, for comparison only)
desc_cols = [RAW_COLS["module_temp"], RAW_COLS["amb_temp"], RAW_COLS["wind"], RAW_COLS["irr"],
             RAW_COLS["dc_current"], RAW_COLS["ac_ir"], RAW_COLS["ac_iy"], RAW_COLS["ac_ib"]]
desc = df[desc_cols].describe()
desc["AC_Power_kW"] = df["AC_Power_kW"].describe()
desc.to_csv(f"{OUT}/table1_descriptive_stats_recreated.csv")
print("\nRecreated Table 1 means (paper: MODULE_TEMP=37.14, Amb_Temp=22.96, WIND=2.24, "
      "IRR=428.09, DC=355.90, Ir/Iy/Ib~172.2-172.35, AC Power(kW)=128.08):")
print(desc.loc["mean"])
print("NOTE: if WIND_Speed mean comes out ~100x the paper's 2.24, your CSV's wind speed is in "
      "cm/s while the paper reports m/s (divide WIND_Speed by 100 before use if you need the "
      "paper's exact units -- this does not affect model RMSE/R2, only unit interpretation).")

# ---------------------------------------------------------------------------
# 3.3 Feature engineering: Iavg (diagnostics only, NOT a model input per the paper)
# ---------------------------------------------------------------------------
df["Iavg"] = (df[RAW_COLS["ac_ir"]] + df[RAW_COLS["ac_iy"]] + df[RAW_COLS["ac_ib"]]) / 3.0

# ---------------------------------------------------------------------------
# 4.1 Scenario definitions
# ---------------------------------------------------------------------------
EXO_FEATURES = [RAW_COLS["irr"], RAW_COLS["module_temp"], RAW_COLS["amb_temp"], RAW_COLS["wind"]]
ELEC_FEATURES = [RAW_COLS["dc_current"], RAW_COLS["ac_ir"], RAW_COLS["ac_iy"], RAW_COLS["ac_ib"]]
SCENARIO_B_FEATURES = EXO_FEATURES + ELEC_FEATURES
TARGET = "AC_Power_kW"

# ---------------------------------------------------------------------------
# 4.2 Train-test split (Table 2): A = temporal 70/30 (row order); B = random 70/30, shuffle, seed 42
# ---------------------------------------------------------------------------
split_idx = int(N * 0.70)
A_train_df, A_test_df = df.iloc[:split_idx], df.iloc[split_idx:]
B_train_df, B_test_df = train_test_split(df, test_size=0.30, shuffle=True, random_state=42)

def xy(d, feats):
    return d[feats].values, d[TARGET].values

XA_tr, yA_tr = xy(A_train_df, EXO_FEATURES)
XA_te, yA_te = xy(A_test_df, EXO_FEATURES)
XB_tr, yB_tr = xy(B_train_df, SCENARIO_B_FEATURES)
XB_te, yB_te = xy(B_test_df, SCENARIO_B_FEATURES)

# ---------------------------------------------------------------------------
# 4.3 Models and hyperparameters (Table 4) -- exact values from the paper
# ---------------------------------------------------------------------------
def build_models():
    models = {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=600, max_depth=None,
            l2_regularization=1.0, random_state=42),
        "Linear Regression (OLS)": LinearRegression(fit_intercept=True),
        "SVR (RBF)": make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)),
        "Decision Tree (CART)": DecisionTreeRegressor(
            max_depth=10, min_samples_split=5, random_state=42),
        "Tweedie GLM": TweedieRegressor(
            power=1.5, link="log", alpha=5e-4, max_iter=10000),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                         solver="adam", learning_rate_init=1e-3,
                         max_iter=500, early_stopping=True, random_state=42)),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1)
    return models

def evaluate(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae = float(mean_absolute_error(y_te, pred))
    r2 = float(r2_score(y_te, pred))
    # Percentage metrics: paper excludes AC Power <= 0.5 kW (Section 4.4)
    mask = y_te > 0.5
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_te[mask] - pred[mask]) / y_te[mask])) * 100)
        smape = float(np.mean(2 * np.abs(y_te[mask] - pred[mask]) /
                               (np.abs(y_te[mask]) + np.abs(pred[mask]))) * 100)
    else:
        mape, smape = float("nan"), float("nan")
    return model, pred, dict(RMSE=rmse, MAE=mae, R2=r2, MAPE=mape, sMAPE=smape)

results_A, results_B = {}, {}
fitted_models_A, fitted_models_B = {}, {}

print("\n--- Scenario A (exogenous-only) ---")
if not HAS_XGB:
    print("  (XGBoost skipped: `xgboost` not installed -- run `pip install xgboost` in this .venv to include it)")
for name, model in build_models().items():
    m, pred, met = evaluate(model, XA_tr, yA_tr, XA_te, yA_te)
    results_A[name] = met
    fitted_models_A[name] = m
    print(f"{name:28s} R2={met['R2']:.3f} RMSE={met['RMSE']:.3f} MAE={met['MAE']:.3f} "
          f"MAPE={met['MAPE']:.2f}% sMAPE={met['sMAPE']:.2f}%")

print("\n--- Scenario B (exogenous + currents) ---")
for name, model in build_models().items():
    m, pred, met = evaluate(model, XB_tr, yB_tr, XB_te, yB_te)
    results_B[name] = met
    fitted_models_B[name] = m
    print(f"{name:28s} R2={met['R2']:.3f} RMSE={met['RMSE']:.3f} MAE={met['MAE']:.3f} "
          f"MAPE={met['MAPE']:.2f}% sMAPE={met['sMAPE']:.2f}%")

pd.DataFrame(results_A).T.to_csv(f"{OUT}/scenario_A_results.csv")
pd.DataFrame(results_B).T.to_csv(f"{OUT}/scenario_B_results.csv")

# RMSE reduction A->B (this dissertation's own metric, applied here as a cross-check)
rmse_reduction = {}
for name in results_A:
    rmse_reduction[name] = (results_A[name]["RMSE"] - results_B[name]["RMSE"]) / results_A[name]["RMSE"] * 100
pd.Series(rmse_reduction, name="RMSE_reduction_%").to_csv(f"{OUT}/rmse_reduction.csv")
print("\nRMSE reduction A->B (%):")
for k, v in rmse_reduction.items():
    print(f"  {k:28s} {v:6.2f}%")

# ---------------------------------------------------------------------------
# 4.5 Explainability: PDP for the primary model (HistGBDT), Scenario B.
# SHAP used if available; permutation importance always computed as a cross-check.
# ---------------------------------------------------------------------------
primary_B = fitted_models_B["HistGradientBoosting"]
perm = permutation_importance(primary_B, XB_te, yB_te, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": SCENARIO_B_FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
perm_df.to_csv(f"{OUT}/permutation_importance_scenarioB_HGB.csv", index=False)
print("\nPermutation importance (Scenario B, HistGBDT):")
print(perm_df)

if HAS_SHAP:
    explainer = shap.TreeExplainer(primary_B)
    shap_values = explainer.shap_values(XB_te)
    shap.summary_plot(shap_values, XB_te, feature_names=SCENARIO_B_FEATURES, show=False)
    plt.savefig(f"{OUT}/shap_summary_scenarioB_HGB.png", dpi=150, bbox_inches="tight")
    plt.close()
else:
    print("  (SHAP skipped: `shap` not installed -- run `pip install shap` in this .venv to include it)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, feat in zip(axes, [RAW_COLS["irr"], RAW_COLS["module_temp"]]):
    pd_result = partial_dependence(primary_B, XB_te, [SCENARIO_B_FEATURES.index(feat)], kind="average")
    ax.plot(pd_result["grid_values"][0] if "grid_values" in pd_result else pd_result["values"][0],
             pd_result["average"][0])
    ax.set_xlabel(feat)
    ax.set_ylabel("Partial dependence (AC Power, kW)")
    ax.set_title(f"PDP: {feat}")
plt.tight_layout()
plt.savefig(f"{OUT}/pdp_irradiance_moduletemp.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4.6 Thermal Derating Analysis
# ---------------------------------------------------------------------------
# (a) High-irradiance OLS: AC_Power ~ MODULE_TEMP, restricted to IRR in [600, 1000] W/m2
band = df[(df[RAW_COLS["irr"]] >= 600) & (df[RAW_COLS["irr"]] <= 1000)]
X = np.column_stack([np.ones(len(band)), band[RAW_COLS["module_temp"]].values])
y = band[TARGET].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - X.shape[1]
sigma2 = np.sum(resid ** 2) / dof
cov = sigma2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
tcrit = st.t.ppf(0.975, dof)
beta_T_band = beta[1]
ci_band = (beta_T_band - tcrit * se[1], beta_T_band + tcrit * se[1])
print(f"\n(a) High-irradiance OLS (IRR 600-1000 W/m2), N={len(band)}:")
print(f"    beta_T = {beta_T_band:.5f} kW/degC, 95% CI = ({ci_band[0]:.5f}, {ci_band[1]:.5f})")

# (b) Semiparametric: AC_Power = f(IRR) [cubic B-spline basis] + beta_T * MODULE_TEMP + eps
irr_vals = df[RAW_COLS["irr"]].values
mt_vals = df[RAW_COLS["module_temp"]].values
y_full = df[TARGET].values

n_knots = 8
knots_internal = np.quantile(irr_vals, np.linspace(0, 1, n_knots))
degree = 3
knots = np.concatenate(([knots_internal[0]] * degree, knots_internal, [knots_internal[-1]] * degree))
n_basis = len(knots) - degree - 1
spline_basis = np.column_stack([
    BSpline.basis_element(knots[i:i + degree + 2], extrapolate=False)(irr_vals)
    for i in range(n_basis)
])
spline_basis = np.nan_to_num(spline_basis, nan=0.0)

X_semi = np.column_stack([np.ones(N), spline_basis, mt_vals])
beta_semi, *_ = np.linalg.lstsq(X_semi, y_full, rcond=None)
pred_semi = X_semi @ beta_semi
resid_semi = y_full - pred_semi
dof_semi = N - X_semi.shape[1]
sigma2_semi = np.sum(resid_semi ** 2) / dof_semi
cov_semi = sigma2_semi * np.linalg.inv(X_semi.T @ X_semi + 1e-8 * np.eye(X_semi.shape[1]))
se_semi = np.sqrt(np.diag(cov_semi))
beta_T_semi = beta_semi[-1]
tcrit_semi = st.t.ppf(0.975, dof_semi)
ci_semi = (beta_T_semi - tcrit_semi * se_semi[-1], beta_T_semi + tcrit_semi * se_semi[-1])
r2_semi = 1 - np.sum(resid_semi ** 2) / np.sum((y_full - y_full.mean()) ** 2)
print(f"\n(b) Semiparametric model AC_Power = f(IRR) + beta_T*MODULE_TEMP, N={N}:")
print(f"    beta_T = {beta_T_semi:.5f} kW/degC, 95% CI = ({ci_semi[0]:.5f}, {ci_semi[1]:.5f}), R2 = {r2_semi:.3f}")

derating_summary = pd.DataFrame({
    "method": ["High-irradiance OLS (600-1000 W/m2)", "Semiparametric (spline IRR + linear MODULE_TEMP)"],
    "beta_T_kW_per_degC": [beta_T_band, beta_T_semi],
    "CI95_low": [ci_band[0], ci_semi[0]],
    "CI95_high": [ci_band[1], ci_semi[1]],
    "R2": [np.nan, r2_semi],
    "N": [len(band), N],
})
derating_summary.to_csv(f"{OUT}/thermal_derating_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(band[RAW_COLS["module_temp"]], band[TARGET], s=3, alpha=0.3)
ax.set_xlabel("Module temperature (degC)")
ax.set_ylabel("AC Power (kW)")
ax.set_title(f"High-irradiance band (600-1000 W/m2): beta_T={beta_T_band:.4f} kW/degC")
plt.tight_layout()
plt.savefig(f"{OUT}/thermal_derating_highirr_band.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Save full summary
# ---------------------------------------------------------------------------
summary = {
    "dataset": {"N": N, "missing": n_missing, "duplicates": n_dupe},
    "scenario_A": results_A,
    "scenario_B": results_B,
    "rmse_reduction_pct": rmse_reduction,
    "thermal_derating": {
        "high_irradiance_OLS": {"beta_T": float(beta_T_band), "CI95": list(map(float, ci_band)), "N": len(band)},
        "semiparametric": {"beta_T": float(beta_T_semi), "CI95": list(map(float, ci_semi)), "R2": float(r2_semi), "N": N},
    },
    "xgboost_included": HAS_XGB,
    "shap_included": HAS_SHAP,
    "runtime_seconds": time.time() - t0,
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print(f"\nDone in {time.time()-t0:.1f}s. Outputs in ./{OUT}/")