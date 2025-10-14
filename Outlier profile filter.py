#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# ============== CONFIG (tweak as needed) ==============
METRICS_CSV = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics.csv'         # input
CLEAN_CSV   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics_clean.csv'   # output: good rows
BAD_CSV     = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics_bad.csv'      # output: flagged rows
REPORT_TXT  = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics_filter_report.txt'

# Hard plausibility window for SOS (DOY) — temperate Europe
SOS_DOY_MIN = 80    # ~ March 1
SOS_DOY_MAX = 140  # ~ July 19

# IQR fence multiplier (1.5 is standard Tukey)
IQR_K = 1.5

# Optional extra checks (set to True to enforce)
ENFORCE_SOS_BEFORE_PEAK = True
# =======================================================

def main():
    df = pd.read_csv(METRICS_CSV)
    if "sos_doy" not in df.columns:
        raise ValueError("metrics CSV must have a 'sos_doy' column.")
    has_peak = "peak_doy" in df.columns

    # Base mask: finite sos
    finite_sos = np.isfinite(df["sos_doy"].astype(float))

    # Rule 1: plausibility window
    in_window = (df["sos_doy"] >= SOS_DOY_MIN) & (df["sos_doy"] <= SOS_DOY_MAX)

    # Rule 2: SOS before peak (if available)
    if has_peak and ENFORCE_SOS_BEFORE_PEAK:
        finite_peak = np.isfinite(df["peak_doy"].astype(float))
        sos_before_peak = df["sos_doy"] < df["peak_doy"]
    else:
        finite_peak = pd.Series(True, index=df.index)
        sos_before_peak = pd.Series(True, index=df.index)

    # Rule 3: robust IQR outlier filter on sos_doy
    sos_valid_vals = df.loc[finite_sos, "sos_doy"].astype(float)
    q1, q3 = sos_valid_vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - IQR_K * iqr
    upper_fence = q3 + IQR_K * iqr
    in_iqr = (df["sos_doy"] >= lower_fence) & (df["sos_doy"] <= upper_fence)

    # Combine “good” mask
    good = finite_sos & in_window & in_iqr & finite_peak & sos_before_peak

    # Build diagnostics for "bad"
    reasons = []
    reasons.append(~finite_sos)                         # 0
    reasons.append(~in_window)                          # 1
    reasons.append(~in_iqr)                             # 2
    reasons.append(~finite_peak)                        # 3
    reasons.append(~sos_before_peak)                    # 4
    reason_labels = ["non_finite_sos", "outside_window", "iqr_outlier",
                     "peak_missing", "sos_after_peak"]

    bad = ~good
    df_bad = df.loc[bad].copy()
    for lab, mask in zip(reason_labels, reasons):
        df_bad[lab] = mask.loc[bad].astype(bool).values

    df_good = df.loc[good].copy()

    # Write outputs
    os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
    df_good.to_csv(CLEAN_CSV, index=False)
    df_bad.to_csv(BAD_CSV, index=False)

    # Simple report
    with open(REPORT_TXT, "w") as f:
        f.write("NDVI metrics filtering report\n")
        f.write(f"Input:  {METRICS_CSV}\n")
        f.write(f"Output (clean): {CLEAN_CSV}\n")
        f.write(f"Output (bad):   {BAD_CSV}\n\n")
        f.write(f"Rows total: {len(df)}\n")
        f.write(f"Kept (good): {good.sum()}\n")
        f.write(f"Removed (bad): {bad.sum()}\n\n")
        f.write(f"SOS window: [{SOS_DOY_MIN}, {SOS_DOY_MAX}]\n")
        f.write(f"IQR fences on sos_doy: [{lower_fence:.2f}, {upper_fence:.2f}]\n")
        if has_peak and ENFORCE_SOS_BEFORE_PEAK:
            f.write("Constraint: sos_doy < peak_doy enforced.\n")
        f.write("\nReasons among bad rows:\n")
        for lab in reason_labels:
            if lab in df_bad.columns:
                f.write(f"  - {lab}: {int(df_bad[lab].sum())}\n")

    print(f"[OK] Clean rows: {len(df_good)}  |  Bad rows: {len(df_bad)}")
    print(f"[OK] Wrote: {CLEAN_CSV}\n[OK] Wrote: {BAD_CSV}\n[OK] Wrote: {REPORT_TXT}")

if __name__ == "__main__":
    main()
