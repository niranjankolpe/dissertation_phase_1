#!/usr/bin/env python3
"""Load + clean + schema-align the NIST Campus Ground Array dataset (Plant B)
to match the Kaggle SolarGeneration schema (Plant A), then profile it.

SOURCE: NIST Campus Photovoltaic (PV) Arrays and Weather Station Data Sets,
Ground Array, Gaithersburg MD, USA. Boyd, M. (2017) "Performance Data from
the NIST Photovoltaic (PV) Arrays and Weather Station," NIST JRES,
DOI: 10.6028/jres.122.040. Public domain (US govt work) - verify exact
Ground-array DC capacity/tilt from the source PDF before writing into the
thesis; this script derives an empirical capacity estimate from the data
itself instead of trusting a secondary summary.

Column mapping used (documented, not guessed):
  MODULE_TEMP         <- SEWSModuleTemp_C_Avg
  Amb_Temp            <- AmbTemp_C_Avg
  WIND_Speed          <- WindSpeedAve_ms   (ALREADY m/s - no /100 needed, unlike Plant A)
  IRR (W/m2)          <- SEWSPOAIrrad_Wm2_Avg   (plane-of-array, tilted - matches Ground mount)
  DC Current in Amps  <- InvIDCin_Avg
  AC Ir/Iy/Ib in Amps <- InvIa_Avg / InvIb_Avg / InvIc_Avg
  AC Power in Watts   <- InvPAC_kW_Avg * 1000

Run: python 7_load_plantB_nist.py --dir data/onemin-Ground-2017
Output: data/plantB_nist_ground_2017.csv (clean, aligned)
        analysis_notes/<ts>_plantB_profile.txt"""
import argparse, datetime, glob, os
import numpy as np, pandas as pd

COLMAP = {
    "SEWSModuleTemp_C_Avg": "MODULE_TEMP",
    "AmbTemp_C_Avg": "Amb_Temp",
    "WindSpeedAve_ms": "WIND_Speed",
    "SEWSPOAIrrad_Wm2_Avg": "IRR (W/m2)",
    "InvIDCin_Avg": "DC Current in Amps",
    "InvIa_Avg": "AC Ir in Amps",
    "InvIb_Avg": "AC Iy in Amps",
    "InvIc_Avg": "AC Ib in Amps",
}
FAULT_COLS = ["InvMainFault_Max", "InvDriveFault_Max", "InvVoltageFault_Max",
              "InvGridFault_Max", "InvTempFault_Max", "InvSystemFault_Max"]
NEEDED_RAW = list(COLMAP) + FAULT_COLS + ["TIMESTAMP", "InvPAC_kW_Avg", "InvOpStatus_Avg"]


def main(a):
    files = sorted(glob.glob(os.path.join(a.dir, "**", "*.csv"), recursive=True))
    print(f"found {len(files)} daily files", flush=True)
    frames = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, usecols=lambda c: c in NEEDED_RAW)
        frames.append(d)
        if (i + 1) % 60 == 0:
            print(f"  loaded {i+1}/{len(files)}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    print("raw rows:", len(df), flush=True)

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=False)
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    raw_n = len(df)
    fault_mask = (df[FAULT_COLS].fillna(0) != 0).any(axis=1)
    n_fault = int(fault_mask.sum())
    df = df[~fault_mask].reset_index(drop=True)

    out = df.rename(columns=COLMAP).copy()
    out["AC Power in Watts"] = df["InvPAC_kW_Avg"] * 1000.0
    keep = ["TIMESTAMP"] + list(COLMAP.values()) + ["AC Power in Watts"]
    out = out[keep]

    n_before_dropna = len(out)
    out = out.dropna().reset_index(drop=True)
    n_after_dropna = len(out)

    # sentinel-glitch filter: a handful of rows (~15/494k, found by inspection) carry
    # sensor-fault sentinel values as large negative numbers (e.g. exactly -982) on
    # power/current columns, not caught by the InvXFault_Max flags. Physical sanity
    # cut, not a tuned threshold: night noise never goes below ~-1 W in this data.
    n_before_sentinel = len(out)
    sane = (out["AC Power in Watts"] > -500) & (out["DC Current in Amps"] > -50)
    out = out[sane].reset_index(drop=True)
    n_sentinel_dropped = n_before_sentinel - len(out)

    os.makedirs("data", exist_ok=True)
    out.to_csv(a.out, index=False)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs("analysis_notes", exist_ok=True)
    prof_path = f"analysis_notes/{ts}_plantB_profile.txt"
    day_capacity_kw = out["AC Power in Watts"].max() / 1000.0
    corr = out[list(COLMAP.values()) + ["AC Power in Watts"]].corr()["AC Power in Watts"].sort_values(ascending=False)

    lines = [
        f"[{ts}] Plant B (NIST Ground Array) load + profile",
        f"Files: {len(files)}  raw rows: {raw_n}",
        f"Rows dropped for active fault code: {n_fault} ({100*n_fault/raw_n:.2f}%)",
        f"Rows dropped for any null after fault filter: {n_before_dropna - n_after_dropna} "
        f"({100*(n_before_dropna-n_after_dropna)/max(n_before_dropna,1):.2f}%)",
        f"Rows dropped for sentinel-glitch values (e.g. -982): {n_sentinel_dropped}",
        f"Final clean rows: {len(out)}",
        f"Date range: {out['TIMESTAMP'].min()} to {out['TIMESTAMP'].max()}",
        f"Empirical peak AC power observed (proxy for DC/AC capacity, NOT nameplate): {day_capacity_kw:.1f} kW",
        "",
        "Correlation of each feature with AC Power in Watts (leakage check, same idea as Plant A):",
        corr.to_string(),
        "",
        "Column ranges:",
        out[list(COLMAP.values()) + ["AC Power in Watts"]].describe().round(3).to_string(),
    ]
    open(prof_path, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten: {a.out}\nwritten: {prof_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/onemin-Ground-2017")
    p.add_argument("--out", default="data/plantB_nist_ground_2017.csv")
    main(p.parse_args())
