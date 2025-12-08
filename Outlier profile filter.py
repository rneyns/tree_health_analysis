#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# ============== CONFIG (tweak as needed) ==============
METRICS_CSV = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/aesculus hippocastanum/ndvi_metrics.csv'         # input
CLEAN_CSV   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/aesculus hippocastanum/ndvi_metrics_clean.csv'   # output: good rows
BAD_CSV     = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/aesculus hippocastanum/ndvi_metrics_bad.csv'      # output: flagged rows
REPORT_TXT  = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/aesculus hippocastanum/ndvi_metrics_filter_report.txt'

# Hard plausibility window for SOS (DOY) — temperate Europe
SOS_DOY_MIN = 80    # ~ March 21
SOS_DOY_MAX = 140   # ~ May 20

# IQR fence multiplier (1.5 is standard Tukey)
IQR_K = 1.5

# Optional extra checks
ENFORCE_SOS_BEFORE_PEAK = True
# =======================================================

def main():
    df = pd.read_csv(METRICS_CSV)

    # Basic column checks
    if "sos_doy" not in df.columns:
        raise ValueError("metrics CSV must have a 'sos_doy' column.")
    if "rmse" not in df.columns:
        raise ValueError("metrics CSV must have an 'rmse' column.")

    has_peak = "peak_doy" in df.columns

    # --------- SOS-based filters ---------
    sos = df["sos_doy"].astype(float)
    finite_sos = np.isfinite(sos)

    # Rule 1: plausibility window
    in_window = (sos >= SOS_DOY_MIN) & (sos <= SOS_DOY_MAX)

    # Rule 2: SOS before peak (if available)
    if has_peak and ENFORCE_SOS_BEFORE_PEAK:
        peak = df["peak_doy"].astype(float)
        finite_peak = np.isfinite(peak)
        sos_before_peak = sos < peak
    else:
        finite_peak = pd.Series(True, index=df.index)
        sos_before_peak = pd.Series(True, index=df.index)

    # Rule 3: robust IQR outlier filter on sos_doy
    sos_valid_vals = sos[finite_sos]
    q1_sos, q3_sos = sos_valid_vals.quantile([0.25, 0.75])
    iqr_sos = q3_sos - q1_sos
    lower_fence_sos = q1_sos - IQR_K * iqr_sos
    upper_fence_sos = q3_sos + IQR_K * iqr_sos
    in_iqr_sos = (sos >= lower_fence_sos) & (sos <= upper_fence_sos)

    # --------- RMSE-based filters ---------
    rmse = df["rmse"].astype(float)
    finite_rmse = np.isfinite(rmse)

    # IQR filter on RMSE (removes extremely bad fits)
    rmse_valid_vals = rmse[finite_rmse]
    q1_rmse, q3_rmse = rmse_valid_vals.quantile([0.25, 0.75])
    iqr_rmse = q3_rmse - q1_rmse
    lower_fence_rmse = q1_rmse - IQR_K * iqr_rmse
    upper_fence_rmse = q3_rmse + IQR_K * iqr_rmse
    in_iqr_rmse = (rmse >= lower_fence_rmse) & (rmse <= upper_fence_rmse)

    # --------- Combine “good” mask ---------
    good = (
        finite_sos &
        in_window &
        in_iqr_sos &
        finite_peak &
        sos_before_peak &
        finite_rmse &
        in_iqr_rmse
    )

    # --------- Diagnostics for "bad" rows ---------
    reasons = []
    reasons.append(~finite_sos)        # 0
    reasons.append(~in_window)         # 1
    reasons.append(~in_iqr_sos)        # 2
    reasons.append(~finite_peak)       # 3
    reasons.append(~sos_before_peak)   # 4
    reasons.append(~finite_rmse)       # 5
    reasons.append(~in_iqr_rmse)       # 6

    reason_labels = [
        "non_finite_sos",
        "outside_sos_window",
        "sos_iqr_outlier",
        "peak_missing",
        "sos_after_peak",
        "non_finite_rmse",
        "rmse_iqr_outlier",
    ]

    bad = ~good
    df_bad = df.loc[bad].copy()
    for lab, mask in zip(reason_labels, reasons):
        df_bad[lab] = mask.loc[bad].astype(bool).values

    df_good = df.loc[good].copy()

    # --------- Write outputs ---------
    os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
    df_good.to_csv(CLEAN_CSV, index=False)
    df_bad.to_csv(BAD_CSV, index=False)

    # --------- Text report ---------
    with open(REPORT_TXT, "w") as f:
        f.write("NDVI metrics filtering report\n")
        f.write(f"Input:  {METRICS_CSV}\n")
        f.write(f"Output (clean): {CLEAN_CSV}\n")
        f.write(f"Output (bad):   {BAD_CSV}\n\n")
        f.write(f"Rows total: {len(df)}\n")
        f.write(f"Kept (good): {int(good.sum())}\n")
        f.write(f"Removed (bad): {int(bad.sum())}\n\n")

        f.write(f"SOS window: [{SOS_DOY_MIN}, {SOS_DOY_MAX}]\n")
        f.write(f"SOS IQR fences: [{lower_fence_sos:.2f}, {upper_fence_sos:.2f}]\n")
        f.write(f"RMSE IQR fences: [{lower_fence_rmse:.4f}, {upper_fence_rmse:.4f}]\n")
        if has_peak and ENFORCE_SOS_BEFORE_PEAK:
            f.write("Constraint: sos_doy < peak_doy enforced.\n")
        f.write("\nReasons among bad rows:\n")
        for lab in reason_labels:
            if lab in df_bad.columns:
                f.write(f"  - {lab}: {int(df_bad[lab].sum())}\n")

    print(f"[OK] Clean rows: {len(df_good)}  |  Bad rows: {len(df_bad)}")
    print(f"[OK] Wrote: {CLEAN_CSV}")
    print(f"[OK] Wrote: {BAD_CSV}")
    print(f"[OK] Wrote: {REPORT_TXT}")

if __name__ == "__main__":
    main()
