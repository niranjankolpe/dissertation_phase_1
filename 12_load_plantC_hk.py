#!/usr/bin/env python3
"""Load + clean + schema-align a THIRD independent PV site (Plant C) for the
"is this a real pattern or just 2 sites?" check the student asked for.

SOURCE: HKUST rooftop PV dataset ("A high-resolution three-year dataset
supporting rooftop photovoltaics (PV) generation analytics"), Sai Kung,
Hong Kong. CC0 (public domain). Zenodo: https://zenodo.org/records/10909062
Paper: Scientific Data, DOI to verify from the Zenodo/journal page before
citing in the thesis - not independently re-verified here beyond the CC0
license text shown on the Zenodo page itself.

Using ONE inverter (LSK North) - a single rooftop array within a 60-station
campus-wide behind-the-meter deployment, sharing ONE campus weather station.
This is structurally different from Plant A/B (each a single dedicated
plant with its own weather station) - note this difference, don't pretend
it's identical in kind, just useful as a third independent check.

IMPORTANT: this dataset has NO module/panel temperature channel (only
ambient air temp from the weather station). So Plant C only supports a
3-feature weather set (Amb_Temp, WIND_Speed, IRR), not the 4-feature set
(+ MODULE_TEMP) used for Plant A vs B. For a fair 3-way comparison, use
the reduced 3-feature set for ALL THREE plants (drop MODULE_TEMP even for
A/B) - see 13_three_site_quick.py, which does this by just not selecting
that column, no need to reload/reclean A or B.

Column mapping:
  Amb_Temp   <- Temperature dataset (deg C)
  WIND_Speed <- Wind dataset (already m/s)
  IRR (W/m2) <- Irradiance dataset
  AC_1/2/3 in Amps <- L1/L2/L3_acCurrent(A)  (3-phase, same structure as A/B)
  AC Power in Watts <- totalActivePower(W)

Run: python 12_load_plantC_hk.py
Output: data/plantC_hk_lsk_north.csv
        analysis_notes/<ts>_plantC_profile.txt"""
import argparse, datetime, io, os, zipfile
import numpy as np, pandas as pd

ZIP = "data/hk_pv_dataset.zip"
INV_PATH = ("Dataset/Time series dataset/PV generation dataset/"
            "PV stations with panel level optimizer/Inverter level dataset/LSK North_Inverter.csv")
MET_DIR = "Dataset/Time series dataset/Meteorological dataset"


def read_csv_from_zip(zf, path, **kw):
    with zf.open(path) as f:
        return pd.read_csv(io.BytesIO(f.read()), **kw)


def main(a):
    zf = zipfile.ZipFile(ZIP)
    inv = read_csv_from_zip(zf, INV_PATH, parse_dates=["Time"])
    inv = inv.rename(columns={"Time": "TIMESTAMP"})
    print(f"inverter raw rows: {len(inv)}", flush=True)

    met = {}
    for name, folder in [("Amb_Temp", "Temperature"), ("WIND_Speed", "Wind"), ("IRR (W/m2)", "Irradiance")]:
        parts = []
        for year in (2021, 2022, 2023):
            path = f"{MET_DIR}/{folder}/{folder.replace(' ', ' ')}_{year}.csv"
            try:
                d = read_csv_from_zip(zf, path, parse_dates=["Time"])
                parts.append(d)
            except KeyError:
                continue
        m = pd.concat(parts, ignore_index=True).rename(columns={"Time": "TIMESTAMP"})
        m = m.sort_values("TIMESTAMP").drop_duplicates("TIMESTAMP")
        met[name] = m
    print("weather files loaded: " + ", ".join(f"{k}={len(v)}" for k, v in met.items()), flush=True)

    inv = inv.sort_values("TIMESTAMP").reset_index(drop=True)
    df = inv[["TIMESTAMP", "totalActivePower(W)", "L1_acCurrent(A)", "L2_acCurrent(A)", "L3_acCurrent(A)"]].copy()
    df = df.rename(columns={"totalActivePower(W)": "AC Power in Watts",
                             "L1_acCurrent(A)": "AC_1 in Amps", "L2_acCurrent(A)": "AC_2 in Amps",
                             "L3_acCurrent(A)": "AC_3 in Amps"})

    for name, m in met.items():
        col = [c for c in m.columns if c != "TIMESTAMP"][0]
        m2 = m[["TIMESTAMP", col]].rename(columns={col: name})
        df = pd.merge_asof(df.sort_values("TIMESTAMP"), m2.sort_values("TIMESTAMP"),
                            on="TIMESTAMP", direction="nearest", tolerance=pd.Timedelta("3min"))

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)

    sane = (df["AC Power in Watts"] >= 0) & (df["AC_1 in Amps"] >= 0) & (df["AC_2 in Amps"] >= 0) & (df["AC_3 in Amps"] >= 0)
    df = df[sane].reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(a.out, index=False)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs("analysis_notes", exist_ok=True)
    corr = df[["Amb_Temp", "WIND_Speed", "IRR (W/m2)", "AC_1 in Amps", "AC_2 in Amps", "AC_3 in Amps",
               "AC Power in Watts"]].corr()["AC Power in Watts"].sort_values(ascending=False)
    lines = [
        f"[{ts}] Plant C (HK LSK North inverter) load + profile",
        f"Rows before dropna/merge-tolerance filter: {n_before}, after: {n_after}, final clean: {len(df)}",
        f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}",
        f"Peak observed power: {df['AC Power in Watts'].max():.0f} W",
        "",
        "Correlation with AC Power in Watts (leakage check, same idea as A/B):",
        corr.to_string(),
        "",
        df.describe().round(3).to_string(),
    ]
    prof_path = f"analysis_notes/{ts}_plantC_profile.txt"
    open(prof_path, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten: {a.out}\nwritten: {prof_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/plantC_hk_lsk_north.csv")
    main(p.parse_args())
