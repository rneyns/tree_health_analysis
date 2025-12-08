"""
Script: test_health_environment_relationships.py

Purpose:
- Test whether any tree health metric is meaningfully correlated with any environmental variable.
- Tests include Pearson, Spearman, Kendall, Partial Correlation, and Distance Correlation.
- Outputs a table summarizing effect sizes + significance with FDR correction.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import pingouin as pg  # pip install pingouin (for partial correlations)
from dcor import distance_correlation  # pip install dcor

# ----------------------------
# USER INPUT
# ----------------------------

df = pd.read_csv('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/ndvi_metrics_with_impervious.csv')

# Define your health metrics
health_vars = [
    "ndvi_peak",
    "ndvi_base",
    "greenup_doy",
    "sos_doy",
    "peak_doy",
    "sen_onset_doy",
    "eos_doy",
    "dormancy_doy",
    "los_days",
    "amplitude",
    "slope_sos_peak",
    "senescence_rate",
    "auc_above_base_full"
]

# Define your environmental predictors
env_vars = [
    "imperv_10m",
    "imperv_20m",
    "poll_no2_anmean",
    "lst_temp_r100",
    "lst_temp_r50",
    "height",
]

# Variable to CONTROL FOR in partial correlations
control_var = "imperv_10m"

# ----------------------------
# RESULTS TABLE
# ----------------------------

results = []

for h in health_vars:
    for e in env_vars:

        if h == e:
            continue

        # Drop NA for the three vars we need
        df_sub = df[[h, e, control_var]].dropna()

        # Convert to numpy arrays
        x = np.asarray(df_sub[h])
        y = np.asarray(df_sub[e])

        # --- Guardrails: make sure both are 1D numeric vectors ---
        # Skip if y (or x) is multi-dimensional, e.g. shape (n, 2)
        if x.ndim != 1 or y.ndim != 1:
            print(f"Skipping pair {h} – {e}: x.shape={x.shape}, y.shape={y.shape} (need 1D)")
            continue

        # Skip if lengths don't match for some reason
        if x.shape[0] != y.shape[0]:
            print(f"Skipping pair {h} – {e}: length mismatch {x.shape[0]} vs {y.shape[0]}")
            continue

        # Try casting to float; skip if it fails (non-numeric)
        try:
            x = x.astype(float)
            y = y.astype(float)
        except ValueError:
            print(f"Skipping pair {h} – {e}: could not convert to float")
            continue

        # -----------------------------
        # Compute correlations
        # -----------------------------
        pear_r, pear_p = pearsonr(x, y)
        spear_rho, spear_p = spearmanr(x, y)
        kend_tau, kend_p = kendalltau(x, y)

        # Partial Spearman: health ~ env | control_var
        pcorr = pg.partial_corr(
            data=df_sub,
            x=h,
            y=e,
            covar=control_var,
            method="spearman"
        )
        part_rho = pcorr["r"].values[0]
        part_p = pcorr["p-val"].values[0]

        # Distance correlation
        dcor_val = distance_correlation(x, y)

        results.append({
            "health_metric": h,
            "env_metric": e,
            "pearson_r": pear_r, "pearson_p": pear_p,
            "spearman_rho": spear_rho, "spearman_p": spear_p,
            "kendall_tau": kend_tau, "kendall_p": kend_p,
            "partial_spearman_rho": part_rho, "partial_p": part_p,
            "distance_corr": dcor_val
        })


# ----------------------------
# CREATE RESULTS DATAFRAME
# ----------------------------

res = pd.DataFrame(results)

# FDR correction across all tests
for col in ["pearson_p", "spearman_p", "kendall_p", "partial_p"]:
    res[col+"_adj"] = multipletests(res[col], method='fdr_bh')[1]

res.to_csv('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/health_environment_stats_summary.csv', index=False)

print("✔ Done! Results saved.")
