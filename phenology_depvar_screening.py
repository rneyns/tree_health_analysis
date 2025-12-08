#!/usr/bin/env python3
"""
Screening analysis for dependent variables (tree-health phenology metrics).

What it does
------------
1) Reads ndvi_metrics.csv produced by your phenology script.
2) Identifies:
   - Phenology metrics (dependent candidates)
   - Stressors (independent) : pm25, impervious_r*, temp_r*, lst_temp*, insolation*
3) Computes Spearman correlations (rho, p, FDR q).
4) Fits two models per phenology metric (with 5-fold CV):
   - Ridge regression (linear, standardized)
   - RandomForestRegressor (nonlinear)
   Reports mean CV R^2 for both, and which is better.
5) Exports:
   - spearman_corr_matrix.csv                  (rows=phenology, cols=stressors)
   - spearman_corr_long.csv                    (tidy table with rho, p, q)
   - spearman_corr_matrix_phenology.csv        (within-phenology correlation)
   - spearman_corr_matrix_stressors.csv        (within-stressors correlation)
   - model_performance.csv                     (per metric CV R^2 and key predictors)
   - plots/corr_heatmap.png                    (full phenology × stressors)
   - plots/corr_heatmap_top_by_category.png    (top metric per phenology category + ndvi_peak)
   - plots/corr_health_within.png              (phenology vs phenology)
   - plots/corr_stressors_within.png           (stressor vs stressor)
   - plots/top_metrics_by_r2.png
"""

import os
import re
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
METRICS_CSV = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/aesculus hippocastanum/ndvi_metrics_clean.csv"
OUT_DIR     = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Results/screening aesculus hippocastanum"

# RF settings
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH    = None
CV_FOLDS        = 5
RANDOM_STATE    = 42
# --------------------------------------------------------------

# Pretty display names for environmental variables (stressors) in plots.
# These act as overrides; anything not here is handled by regex-based fallbacks.
STRESSOR_LABELS = {
    "height": "Tree height (m)",
    "area": "Crown area (m²)",
    "poll_pm25": "PM₂.₅ concentration",

    "impervious_r_20": "Imperviousness (20 m)",
    "impervious_r_50": "Imperviousness (50 m)",
    "impervious_r_100": "Imperviousness (100 m)",
    "impervious_r_200": "Imperviousness (200 m)",

    "temp_r_20": "Air temperature (20 m)",
    "temp_r_50": "Air temperature (50 m)",
    "temp_r_100": "Air temperature (100 m)",
    "temp_r_200": "Air temperature (200 m)",

    "lst_temp_r_20": "LST (20 m)",
    "lst_temp_r_50": "LST (50 m)",
    "lst_temp_r_100": "LST (100 m)",
    "lst_temp_r_200": "LST (200 m)",

    "insolation_mean": "Solar insolation (mean)",
    "insolation_max": "Solar insolation (max)",
}

# Pretty display names for phenology / health metrics
PHENOLOGY_LABELS = {
    "sos_doy": "Start of season (DOY)",
    "peak_doy": "Peak NDVI (DOY)",
    "eos_doy": "End of season (DOY)",
    "sen_onset_doy": "Onset of senescence (DOY)",
    "los_days": "Length of season (days)",

    "ndvi_base": "Base NDVI",
    "ndvi_peak": "Peak NDVI",
    "ndvi_eos": "NDVI at end of season",
    "ndvi_sen_onset": "NDVI at senescence onset",
    "amplitude": "Seasonal NDVI amplitude",

    "slope_sos_peak": "Green-up rate",
    "senescence_rate": "Senescence rate",
    "mean_senescence_rate": "Mean senescence rate",
    "asymmetry": "Seasonal asymmetry",
    "plateau_days": "Plateau length (days)",
    "decline_days": "Decline length (days)",

    "auc_full": "AUC (full season)",
    "auc_sos_eos": "AUC (SoS–EoS)",
    "auc_above_base_full": "AUC above base (full)",
    "auc_above_base_sos_eos": "AUC above base (SoS–EoS)",

    "rmse": "Model RMSE",
    "resid_iqr": "Residual IQR",
}

# Phenology categories for “top-per-category” heatmap
PHENOLOGY_CATEGORIES = {
    "timing": [
        "sos_doy", "peak_doy", "eos_doy", "sen_onset_doy", "los_days"
    ],
    "magnitude": [
        "ndvi_base", "ndvi_peak", "ndvi_eos", "ndvi_sen_onset", "amplitude"
    ],
    "shape": [
        "slope_sos_peak", "senescence_rate", "mean_senescence_rate",
        "asymmetry", "plateau_days", "decline_days"
    ],
    "integral": [
        "auc_full", "auc_sos_eos",
        "auc_above_base_full", "auc_above_base_sos_eos"
    ],
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ---------- Pretty name helpers ----------

def pretty_stressor_name(col: str) -> str:
    """Return a human-readable label for a stressor column."""
    if col in STRESSOR_LABELS:
        return STRESSOR_LABELS[col]

    # Imperviousness buffers: impervious_r20, impervious_r_20, etc.
    m = re.match(r"impervious_r_?(\d+)", col)
    if m:
        return f"Imperviousness ({m.group(1)} m)"

    # Air temp buffers: temp_r20, temp_r_20, etc.
    m = re.match(r"temp_r_?(\d+)", col)
    if m:
        return f"Air temperature ({m.group(1)} m)"

    # LST buffers: lst_temp_r20, lst_temp_r_20, etc.
    m = re.match(r"lst_temp_r_?(\d+)", col)
    if m:
        return f"LST ({m.group(1)} m)"

    # Insolation: insolation_mean, insolation_max, insolation_xxx
    m = re.match(r"insolation_(.+)", col)
    if m:
        suffix = m.group(1).replace("_", " ")
        return f"Solar insolation ({suffix})"

    # Pollution: poll_xxx
    m = re.match(r"poll_(.+)", col)
    if m:
        pollutant = m.group(1).upper().replace("_", " ")
        return f"{pollutant} concentration"

    # Fallback: generic prettification
    return col.replace("_", " ").title()


def pretty_phenology_name(metric: str) -> str:
    """Return a human-readable label for a phenology/health metric."""
    if metric in PHENOLOGY_LABELS:
        return PHENOLOGY_LABELS[metric]
    return metric.replace("_", " ").title()


# ---------- Column detection ----------

def find_columns(df):
    """Identify phenology metrics (dependent candidates) and stressor columns."""
    # Stressors
    stress_poll = [c for c in df.columns if c.startswith("poll")]
    stress_imperv = [c for c in df.columns if c.startswith("impervious_r")]
    stress_temp   = [c for c in df.columns if c.startswith("temp_r")]
    lst_stress_temp = [c for c in df.columns if c.startswith("lst_temp")]
    stress_insolation = [c for c in df.columns if c.startswith("insolation")]
    stressors = (
        ['height', 'area'] +
        sorted(stress_insolation) +
        sorted(lst_stress_temp) +
        (stress_poll) +
        sorted(stress_imperv) +
        sorted(stress_temp)
    )

    # Phenology metrics (common names from your script)
    phenology_candidates = [
        "sos_doy", "peak_doy", "eos_doy", "sen_onset_doy", "los_days",
        "ndvi_base", "ndvi_peak", "ndvi_eos", "ndvi_sen_onset", "amplitude",
        "slope_sos_peak", "senescence_rate", "mean_senescence_rate", "asymmetry",
        "plateau_days", "decline_days",
        "auc_full", "auc_sos_eos", "auc_above_base_full", "auc_above_base_sos_eos",
        "rmse", "resid_iqr",
    ]
    phenology = [c for c in phenology_candidates if c in df.columns]

    # Drop obviously non-numeric
    stressors = [c for c in stressors if pd.api.types.is_numeric_dtype(df[c])]
    phenology = [c for c in phenology if pd.api.types.is_numeric_dtype(df[c])]

    return phenology, stressors


# ---------- Correlations ----------

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
    # FDR correction (global)
    mask = long["p"].notna()
    if mask.any():
        _, q, _, _ = multipletests(long.loc[mask, "p"], alpha=0.05, method="fdr_bh")
        long.loc[mask, "q"] = q
    else:
        long["q"] = np.nan

    return rho_mat, p_mat, long


def spearman_within(df, cols, min_n=10):
    """
    Compute a Spearman correlation matrix within a set of columns.
    Uses pandas' corr(method='spearman') for robustness.
    Only numeric columns are kept. Pairs with < min_n complete obs
    are set to NaN.
    """
    numeric_cols = [
        c for c in cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    data = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    corr = data.corr(method="spearman")

    if min_n is not None and min_n > 1:
        for c1 in numeric_cols:
            for c2 in numeric_cols:
                m = data[[c1, c2]].dropna()
                if len(m) < min_n:
                    corr.loc[c1, c2] = np.nan

    return corr


# ---------- Modelling ----------

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

    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=15, random_state=random_state, n_jobs=-1)
    imp_series = pd.Series(perm.importances_mean, index=Xcols).sort_values(ascending=False)
    top5 = list(imp_series.head(5).index)
    top5_imp = [float(imp_series[c]) for c in top5]
    return mean_r2, top5, top5_imp


# ---------- Selection for compressed heatmap ----------

def select_top_metrics_by_category(rho_mat, categories, always_include=None):
    """
    For each phenology category, pick the metric with the highest
    absolute Spearman correlation (across all stressors).
    Additionally include any metrics listed in always_include (if present).

    Returns:
      - list of metric names (row order)
      - list of labels with pretty health metric name + category
    """
    if always_include is None:
        always_include = []

    # Reverse mapping: metric -> category
    metric_to_cat = {}
    for cat, metrics in categories.items():
        for m in metrics:
            metric_to_cat[m] = cat

    selected_metrics = []
    labels = []

    # 1) Top-per-category selection
    for cat, metrics in categories.items():
        available = [m for m in metrics if m in rho_mat.index]
        if not available:
            continue

        sub = rho_mat.loc[available].abs()
        max_per_metric = sub.max(axis=1)
        best_metric = max_per_metric.idxmax()

        if best_metric not in selected_metrics:
            selected_metrics.append(best_metric)
            pretty_name = pretty_phenology_name(best_metric)
            labels.append(f"{pretty_name} ({cat})")

    # 2) Force-include requested metrics (e.g. ndvi_peak),
    #    but label them by their category (e.g. magnitude), not "forced include".
    for m in always_include:
        if m in rho_mat.index and m not in selected_metrics:
            selected_metrics.append(m)
            cat = metric_to_cat.get(m, "magnitude" if m == "ndvi_peak" else "other")
            pretty_name = pretty_phenology_name(m)
            labels.append(f"{pretty_name} ({cat})")

    return selected_metrics, labels


# ---------- Plotting ----------

def plot_heatmap(matrix, row_labels, col_labels, title, out_path,
                 vmin=None, vmax=None):
    """Generic helper to plot a colored correlation matrix."""
    plt.figure(figsize=(max(6, 0.4 * len(col_labels) + 2),
                        max(6, 0.4 * len(row_labels) + 2)))
    plt.imshow(matrix, aspect="auto", interpolation="nearest",
               vmin=vmin, vmax=vmax)
    plt.colorbar(label="Spearman rho")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xticks(range(len(col_labels)), col_labels, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def quick_plots(rho_mat, perf_df, phenology_rho_within, stressor_rho_within, out_dir):
    ensure_dir(out_dir)

    # ---- Full phenology vs stressors heatmap (absolute |rho|) ----
    H = rho_mat.abs()
    xlabels = [pretty_stressor_name(c) for c in H.columns]
    ylabels = [pretty_phenology_name(m) for m in H.index]

    plot_heatmap(
        H.values,
        ylabels,
        xlabels,
        "Phenology vs Stressors — |Spearman rho| (full)",
        os.path.join(out_dir, "corr_heatmap.png"),
        vmin=0.0, vmax=1.0
    )

    # ---- Compressed heatmap: top metric per category + ndvi_peak ----
    top_rows, top_labels = select_top_metrics_by_category(
        rho_mat,
        PHENOLOGY_CATEGORIES,
        always_include=["ndvi_peak"]
    )
    if top_rows:
        H2 = rho_mat.loc[top_rows].abs()
        xlabels2 = [pretty_stressor_name(c) for c in H2.columns]

        plot_heatmap(
            H2.values,
            top_labels,
            xlabels2,
            "Phenology vs Stressors — top per category + ndvi_peak",
            os.path.join(out_dir, "corr_heatmap_top_by_category.png"),
            vmin=0.0, vmax=1.0
        )

    # ---- NEW: within-phenology (health metrics) correlation matrix ----
    if phenology_rho_within is not None and not phenology_rho_within.empty:
        yx_labels = [pretty_phenology_name(m) for m in phenology_rho_within.index]
        plot_heatmap(
            phenology_rho_within.values,
            yx_labels,
            yx_labels,
            "Correlation between health/phenology metrics (Spearman rho)",
            os.path.join(out_dir, "corr_health_within.png"),
            vmin=-1.0, vmax=1.0
        )

    # ---- NEW: within-stressor (environmental) correlation matrix ----
    if stressor_rho_within is not None and not stressor_rho_within.empty:
        yx_labels = [pretty_stressor_name(s) for s in stressor_rho_within.index]
        plot_heatmap(
            stressor_rho_within.values,
            yx_labels,
            yx_labels,
            "Correlation between environmental variables (Spearman rho)",
            os.path.join(out_dir, "corr_stressors_within.png"),
            vmin=-1.0, vmax=1.0
        )

    # ---- Bar chart: top 10 phenology metrics by best CV R^2 ----
    top = perf_df.sort_values("best_cv_r2", ascending=False).head(10)
    pretty_y = [pretty_phenology_name(m) for m in top["phenology"]]

    plt.figure(figsize=(8, max(4, 0.4 * len(top))))
    plt.barh(pretty_y, top["best_cv_r2"])
    for i, v in enumerate(top["best_cv_r2"].values):
        plt.text(v + 0.005, i, f"{v:.2f}", va="center")
    plt.gca().invert_yaxis()
    plt.xlabel("Best CV R²")
    plt.title("Top phenology metrics by explainability (best of Ridge/RF)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_metrics_by_r2.png"), dpi=180)
    plt.close()


# ---------- Main ----------

def main():
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "plots"))

    df = pd.read_csv(METRICS_CSV)

    phenology, stressors = find_columns(df)
    if not phenology:
        raise SystemExit("No phenology columns found in your metrics CSV.")
    if not stressors:
        raise SystemExit("No stressor columns (pm25 / impervious_r* / temp_r* / lst_temp* / insolation*) found.")

    print(f"[INFO] Phenology metrics ({len(phenology)}): {phenology}")
    print(f"[INFO] Stressors ({len(stressors)}): {stressors}")

    # A) Correlation matrices within phenology and within stressors
    phenology_rho_within = spearman_within(df, phenology)
    phenology_rho_within.to_csv(os.path.join(OUT_DIR, "spearman_corr_matrix_phenology.csv"))

    stressor_rho_within = spearman_within(df, stressors)
    stressor_rho_within.to_csv(os.path.join(OUT_DIR, "spearman_corr_matrix_stressors.csv"))

    # 1) Spearman correlations (phenology vs stressors)
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
            "n_complete": int(df[[ycol] + stressors].replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
            "ridge_cv_r2": round(float(ridge_r2), 4) if not np.isnan(ridge_r2) else np.nan,
            "rf_cv_r2": round(float(rf_r2), 4) if not np.isnan(rf_r2) else np.nan,
            "best_model": best_model,
            "best_cv_r2": round(float(best_r2), 4) if not np.isnan(best_r2) else np.nan,
            "best_top_predictors": ";".join(best_top),
            "best_top_values": ";".join([f"{v:.4f}" for v in best_vals]),
        })

    perf_df = pd.DataFrame(perf_rows).sort_values("best_cv_r2", ascending=False)
    perf_df.to_csv(os.path.join(OUT_DIR, "model_performance.csv"), index=False)

    # 3) Quick plots (now includes within-phenology and within-stressor heatmaps)
    quick_plots(rho_mat, perf_df, phenology_rho_within, stressor_rho_within,
                os.path.join(OUT_DIR, "plots"))

    # 4) Friendly console summary
    print("\n=== Top phenology metrics by explainability (best CV R²) ===")
    print(perf_df.head(10).to_string(index=False))

    print(f"\n[OK] Wrote:"
          f"\n - {os.path.join(OUT_DIR, 'spearman_corr_matrix.csv')}"
          f"\n - {os.path.join(OUT_DIR, 'spearman_corr_long.csv')}"
          f"\n - {os.path.join(OUT_DIR, 'spearman_corr_matrix_phenology.csv')}"
          f"\n - {os.path.join(OUT_DIR, 'spearman_corr_matrix_stressors.csv')}"
          f"\n - {os.path.join(OUT_DIR, 'model_performance.csv')}"
          f"\n - plots/corr_heatmap.png"
          f"\n - plots/corr_heatmap_top_by_category.png"
          f"\n - plots/corr_health_within.png"
          f"\n - plots/corr_stressors_within.png"
          f"\n - plots/top_metrics_by_r2.png")


if __name__ == "__main__":
    main()
