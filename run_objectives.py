import os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tiny_transformer import TinyTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
EXO = ["MODULE_TEMP", "Amb_Temp", "WIND_Speed", "IRR (W/m2)"]
ELE = ["DC Current in Amps", "AC Ir in Amps", "AC Iy in Amps", "AC Ib in Amps"]
Y = "AC Power in Watts"
PLANTS = {
    "A Hassan": ("Generation_data.csv", False),
    "B Ground": ("data/plantB_nist_ground_2017.csv", True),
    # "B Roof": ("data/plantB_nist_roof_2017.csv", True),
    # "B Canopy": ("data/plantB_nist_canopy_2017.csv", True),
}
MODELS = {
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(learning_rate=0.06, max_iter=600, l2_regularization=1.0, random_state=42),
    "XGBoost": lambda: XGBRegressor(n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    "Linear": lambda: LinearRegression(),
    "SVR": lambda: SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
    "Tweedie GLM": lambda: TweedieRegressor(power=1.5, link="log", alpha=5e-4, max_iter=10000, solver="newton-cholesky"),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=42)),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Transformer": lambda: TinyTransformer(seed=0, epochs=100),
}
lag = lambda cols: [c + "_lag" for c in cols]
CASES = {
    ("Same instant", "Exogenous"): EXO,
    ("Same instant", "Exogenous + Electric"): EXO + ELE,
    ("Lagged 5 min", "Exogenous"): lag(EXO),
    ("Lagged 5 min", "Exogenous + Electric"): lag(EXO + ELE),
}


def load(path, stamped):
    d = pd.read_csv(path) if os.path.exists(path) else pd.read_excel(path.replace(".csv", ".xlsx"))
    if stamped:
        t = pd.to_datetime(d["TIMESTAMP"], utc=True).dt.floor("5min")
        d = d[EXO + ELE + [Y]].groupby(t).mean()
        prev = d[EXO + ELE].reindex(d.index - pd.Timedelta(minutes=5))
        prev.index = d.index
        d = d.join(prev.add_suffix("_lag"))
    d = d.dropna()
    d = d[d["IRR (W/m2)"] > 10]
    n = int(0.7 * len(d))
    return d.iloc[:n], d.iloc[n:]


def scores(model, tr, te, cols):
    X, y, t = tr[cols].values, tr[Y].values / 1000, te[Y].values / 1000
    p = MODELS[model]().fit(X, y).predict(te[cols].values)
    return {"RMSE_kW": np.sqrt(np.mean((t - p) ** 2)), "MAE_kW": np.mean(np.abs(t - p)), "R2": r2_score(t, p)}, t, p


TITLE = {"A Hassan": "Dataset 1 (Hassan)", "B Ground": "Dataset 2 (NIST Ground)", "B Roof": "Dataset 3 (NIST Roof)", "B Canopy": "Dataset 4 (NIST Canopy)"}
NAME = {"HistGradientBoosting": "HistGBDT", "Linear": "Linear Regression", "SVR": "SVR (RBF)"}
PANEL = {"Same instant": "Contemporaneous", "Lagged 5 min": "Lagged (5 min)"}
SCENARIO = dict(zip(CASES, "ABCD"))
SCATTER_MODEL = "HistGradientBoosting"
BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#222222", "#d9d9d9"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.6, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK})
os.makedirs("figures", exist_ok=True)


def fig_rmse(plant, df):
    timings = [t for t in PANEL if t in set(df["Timing"])]
    models = [m for m in MODELS if m in set(df["Model"])][::-1]
    fig, axes = plt.subplots(1, len(timings), figsize=(3.5, 2.5), sharex=True, sharey=True, squeeze=False)
    for ax, timing in zip(axes[0], timings):
        d = df[df["Timing"] == timing].pivot(index="Model", columns="Inputs", values="RMSE_kW").loc[models]
        y = np.arange(len(models))
        ax.hlines(y, d["Exogenous + Electric"], d["Exogenous"], color="#9a9a9a", lw=1.0, zorder=2)
        ax.scatter(d["Exogenous"], y, s=20, marker="o", color=BLUE, zorder=3, label="Exogenous-only")
        ax.scatter(d["Exogenous + Electric"], y, s=20, marker="s", color=ORANGE, zorder=3, label="Exogenous + Currents")
        ax.set_title(PANEL[timing], fontsize=8)
        ax.set_xlim(0, df["RMSE_kW"].max() * 1.07)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_xlabel("Test RMSE (kW)")
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.tick_params(length=2)
        ax.set_yticks(y)
        ax.set_yticklabels([NAME.get(m, m) for m in models])
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.8)
    fig.savefig(f"figures/Fig_RMSE_{plant.replace(' ', '_')}.png", dpi=600)
    plt.close(fig)


def fig_predicted(plant, preds):
    n = len(preds) // 2
    fig, axes = plt.subplots(n, 2, figsize=(3.5, 1.7 * n + 0.35), sharex=True, sharey=True, squeeze=False)
    top = max(max(t.max(), p.max()) for t, p in preds.values())
    for ax, (case, (t, p)) in zip(axes.ravel(), preds.items()):
        ax.scatter(t, p, s=1.5, color=BLUE, alpha=0.2, linewidths=0, rasterized=True, zorder=2)
        ax.plot([0, top], [0, top], color=INK, lw=0.7, zorder=3)
        ax.set_title(f"Scenario {SCENARIO[case]}", fontsize=8)
        ax.text(0.04, 0.95, f"RMSE {np.sqrt(np.mean((t - p) ** 2)):.2f} kW", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.5, zorder=0)
        ax.tick_params(length=2)
    for ax in axes[-1]:
        ax.set_xlabel("Measured (kW)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted (kW)")
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    fig.savefig(f"figures/Fig_Predicted_{NAME[SCATTER_MODEL]}_{plant.replace(' ', '_')}.png", dpi=600)
    plt.close(fig)


rows = []
for plant, (path, stamped) in PLANTS.items():
    tr, te = load(path, stamped)
    preds = {}
    for model in MODELS:
        for case, cols in CASES.items():
            if stamped or case[0] == "Same instant":
                m, t, p = scores(model, tr, te, cols)
                rows.append({"Plant": plant, "Model": model, "Timing": case[0], "Inputs": case[1], **m})
                pd.DataFrame(rows).round(4).to_csv("Dissertation_Objectives_Results.csv", index=False)
                if model == SCATTER_MODEL:
                    preds[case] = (t, p)
    df = pd.DataFrame(rows)
    fig_rmse(plant, df[df["Plant"] == plant])
    fig_predicted(plant, preds)
