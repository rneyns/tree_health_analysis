#!/usr/bin/env python3
"""
Screening analysis for dependent variables (tree-health phenology metrics).

What it does
------------
1) Reads ndvi_metrics.csv produced by your phenology script.
2) Identifies:
   - Phenology metrics (dependent candidates)
   - Stressors (independent) : pm25, impervious_r*, temp_r*
3) Computes Spearman correlations (rho, p, FDR q).
4) Fits two models per phenology metric (with 5-fold CV):
   - Ridge regression (linear, standardized)
   - RandomForestRegressor (nonlinear)
   Reports mean CV R^2 for both, and which is better.
5) Exports:
   - spearman_corr_matrix.csv  (rows=phenology, cols=stressors)
   - spearman_corr_long.csv    (tidy table with rho, p, q)
   - model_performance.csv     (per metric CV R^2 and key predictors)
   - plots/corr_heatmap.png, plots/top_metrics_by_r2.png

Edit the CONFIG paths below and run.
"""

import os
import re
import math
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance

# --------------------------- CONFIG ---------------------------
METRICS_CSV = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/acer pseudoplatanus/ndvi_metrics_clean.csv"
OUT_DIR     = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Results/screening"

# RF settings
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH    = None
CV_FOLDS        = 5
RANDOM_STATE    = 42
# --------------------------------------------------------------


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def find_columns(df):
    """Identify phenology metrics (dependent candidates) and stressor columns."""
    # Stressors
    stress_poll = [c for c in df.columns if c.startswith("poll")]
    stress_imperv = [c for c in df.columns if c.startswith("impervious_r")]
    stress_temp   = [c for c in df.columns if c.startswith("temp_r")]
    lst_stress_temp = [c for c in df.columns if c.startswith("lst_temp")]
    stress_insolation = [c for c in df.columns if c.startswith("insolation")]
    stressors = ['height','area'] + sorted(stress_insolation) + sorted(lst_stress_temp) + (stress_poll) + sorted(stress_imperv) + sorted(stress_temp)

    # Phenology metrics (common names from your script)
    phenology_candidates = [
        "sos_doy","peak_doy","eos_doy","sen_onset_doy","los_days",
        "ndvi_base","ndvi_peak","ndvi_eos","ndvi_sen_onset","amplitude",
        "slope_sos_peak","senescence_rate","mean_senescence_rate","asymmetry",
        "plateau_days","decline_days",
        "auc_full","auc_sos_eos","auc_above_base_full","auc_above_base_sos_eos",
        "rmse","resid_iqr",
    ]
    phenology = [c for c in phenology_candidates if c in df.columns]

    # Drop obviously non-numeric
    stressors = [c for c in stressors if pd.api.types.is_numeric_dtype(df[c])]
    phenology = [c for c in phenology if pd.api.types.is_numeric_dtype(df[c])]

    return phenology, stressors


def spearman_matrix(df, phenology, stressors):
    """Compute Spearman rho, p for each phenology vs each stressor; return matrix and tidy long table with FDR."""
    rows = []
    rho_mat = pd.DataFrame(index=phenology, columns=stressors, dtype=float)
    p_mat   = pd.DataFrame(index=phenology, columns=stressors, dtype=float)

    for ycol in phenology:
        for xcol in stressors:
            m = df[[ycol, xcol]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(m) < 10:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(m[ycol], m[xcol])
            rho_mat.loc[ycol, xcol] = rho
            p_mat.loc[ycol, xcol] = p
            rows.append({"phenology": ycol, "stressor": xcol, "rho": rho, "p": p, "n": len(m)})

    long = pd.DataFrame(rows)
    # FDR correction within each phenology (optional, here global)
    mask = long["p"].notna()
    if mask.any():
        _, q, _, _ = multipletests(long.loc[mask, "p"], alpha=0.05, method="fdr_bh")
        long.loc[mask, "q"] = q
    else:
        long["q"] = np.nan

    return rho_mat, p_mat, long


def cv_r2_ridge(df, ycol, Xcols, random_state=RANDOM_STATE, cv_folds=CV_FOLDS):
    m = df[[ycol] + Xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(m) < (cv_folds + 5):  # need enough samples
        return np.nan, [], []
    y = m[ycol].values
    X = m[Xcols].values

    alphas = np.logspace(-3, 3, 25)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=alphas, cv=cv_folds))
    ])
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    mean_r2 = float(np.nanmean(scores))

    # Fit once on full data to inspect coefficients
    pipe.fit(X, y)
    coef = pipe.named_steps["ridge"].coef_
    coef_series = pd.Series(coef, index=Xcols).sort_values(key=np.abs, ascending=False)
    top5 = list(coef_series.head(5).index)
    top5_coef = [float(coef_series[c]) for c in top5]
    return mean_r2, top5, top5_coef


def cv_r2_rf(df, ycol, Xcols, random_state=RANDOM_STATE, cv_folds=CV_FOLDS):
    m = df[[ycol] + Xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(m) < (cv_folds + 5):
        return np.nan, [], []
    y = m[ycol].values
    X = m[Xcols].values

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=random_state,
        n_jobs=-1,
        oob_score=False
    )
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
    mean_r2 = float(np.nanmean(scores))

    # Fit once for permutation importance (more robust than Gini)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=15, random_state=random_state, n_jobs=-1)
    imp_series = pd.Series(perm.importances_mean, index=Xcols).sort_values(ascending=False)
    top5 = list(imp_series.head(5).index)
    top5_imp = [float(imp_series[c]) for c in top5]
    return mean_r2, top5, top5_imp

def spearman_within(df, cols, min_n=10):
    """
    Compute a Spearman correlation matrix within a set of columns.
    Uses pandas' corr(method='spearman') for robustness.
    Only numeric columns are kept. Pairs with < min_n complete obs
    are set to NaN.
    """
    # Keep only numeric columns that exist
    numeric_cols = [
        c for c in cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    # Clean infinities, keep only those columns
    data = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # First compute full Spearman correlation with pandas (safe)
    corr = data.corr(method="spearman")

    # Enforce minimum n per pair (optional but close to what you wanted)
    if min_n is not None and min_n > 1:
        for c1 in numeric_cols:
            for c2 in numeric_cols:
                m = data[[c1, c2]].dropna()
                if len(m) < min_n:
                    corr.loc[c1, c2] = np.nan

    return corr



def quick_plots(rho_mat, perf_df, out_dir):
    ensure_dir(out_dir)
    # Heatmap of absolute Spearman rho
    H = rho_mat.abs()
    plt.figure(figsize=(max(6, 0.35*H.shape[1]), max(6, 0.35*H.shape[0])))
    plt.imshow(H.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="|Spearman rho|")
    plt.yticks(range(len(H.index)), H.index)
    plt.xticks(range(len(H.columns)), H.columns, rotation=90)
    plt.title("Phenology vs Stressors — |Spearman|")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_heatmap.png"), dpi=180)
    plt.close()

    # Bar chart: top 10 phenology metrics by best CV R^2
    top = perf_df.sort_values("best_cv_r2", ascending=False).head(10)
    plt.figure(figsize=(8, max(4, 0.4*len(top))))
    plt.barh(top["phenology"], top["best_cv_r2"])
    for i, v in enumerate(top["best_cv_r2"].values):
        plt.text(v + 0.005, i, f"{v:.2f}", va="center")
    plt.gca().invert_yaxis()
    plt.xlabel("Best CV R²")
    plt.title("Top phenology metrics by explainability (best of Ridge/RF)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_metrics_by_r2.png"), dpi=180)
    plt.close()


def main():
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "plots"))

    df = pd.read_csv(METRICS_CSV)

    phenology, stressors = find_columns(df)
    if not phenology:
        raise SystemExit("No phenology columns found in your metrics CSV.")
    if not stressors:
        raise SystemExit("No stressor columns (pm25 / impervious_r* / temp_r*) found.")

    print(f"[INFO] Phenology metrics ({len(phenology)}): {phenology}")
    print(f"[INFO] Stressors ({len(stressors)}): {stressors}")

    # A) Correlation matrices within phenology and within stressors
    phenology_rho_within = spearman_within(df, phenology)
    phenology_rho_within.to_csv(os.path.join(OUT_DIR, "spearman_corr_matrix_phenology.csv"))

    stressor_rho_within = spearman_within(df, stressors)
    stressor_rho_within.to_csv(os.path.join(OUT_DIR, "spearman_corr_matrix_stressors.csv"))


    # 1) Spearman correlations
    rho_mat, p_mat, long = spearman_matrix(df, phenology, stressors)
    rho_mat.to_csv(os.path.join(OUT_DIR, "spearman_corr_matrix.csv"))
    long.sort_values(["phenology", "q", "p", "rho"], inplace=True, na_position="last")
    long.to_csv(os.path.join(OUT_DIR, "spearman_corr_long.csv"), index=False)

    # 2) Screening models per phenology metric
    perf_rows = []
    for ycol in phenology:
        ridge_r2, ridge_top, ridge_vals = cv_r2_ridge(df, ycol, stressors)
        rf_r2, rf_top, rf_vals = cv_r2_rf(df, ycol, stressors)

        if np.isnan(ridge_r2) and np.isnan(rf_r2):
            best_model, best_r2 = None, np.nan
            best_top, best_vals = [], []
        elif (ridge_r2 if not np.isnan(ridge_r2) else -np.inf) >= (rf_r2 if not np.isnan(rf_r2) else -np.inf):
            best_model, best_r2 = "ridge", ridge_r2
            best_top, best_vals = ridge_top, ridge_vals
        else:
            best_model, best_r2 = "rf", rf_r2
            best_top, best_vals = rf_top, rf_vals

        perf_rows.append({
            "phenology": ycol,
            "n_complete": int(df[[ycol] + stressors].replace([np.inf,-np.inf], np.nan).dropna().shape[0]),
            "ridge_cv_r2": round(float(ridge_r2), 4) if not np.isnan(ridge_r2) else np.nan,
            "rf_cv_r2": round(float(rf_r2), 4) if not np.isnan(rf_r2) else np.nan,
            "best_model": best_model,
            "best_cv_r2": round(float(best_r2), 4) if not np.isnan(best_r2) else np.nan,
            "best_top_predictors": ";".join(best_top),
            "best_top_values": ";".join([f"{v:.4f}" for v in best_vals]),
        })

    perf_df = pd.DataFrame(perf_rows).sort_values("best_cv_r2", ascending=False)
    perf_df.to_csv(os.path.join(OUT_DIR, "model_performance.csv"), index=False)

    # 3) Quick plots
    quick_plots(rho_mat, perf_df, os.path.join(OUT_DIR, "plots"))

    # 4) Friendly console summary
    print("\n=== Top phenology metrics by explainability (best CV R²) ===")
    print(perf_df.head(10).to_string(index=False))

    print(f"\n[OK] Wrote:\n - {os.path.join(OUT_DIR, 'spearman_corr_matrix.csv')}"
          f"\n - {os.path.join(OUT_DIR, 'spearman_corr_long.csv')}"
          f"\n - {os.path.join(OUT_DIR, 'model_performance.csv')}"
          f"\n - plots/corr_heatmap.png"
          f"\n - plots/top_metrics_by_r2.png")


if __name__ == "__main__":
    main()
