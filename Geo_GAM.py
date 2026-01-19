#!/usr/bin/env python3
"""
Run the SAME (non-spatial) GAM workflow for MULTIPLE species CSVs.

For each species:
- Load CSV
- Clean/ensure numeric
- Optional biological filters
- Optional z-score standardization (recommended for comparing effect sizes)
- Fit GAMs: health ~ s(pred) + s(height) (+ optional extra controls)
- Quantify "importance" of each predictor for each health metric using
  Δ explained deviance (pseudo-R²) when dropping that predictor:
      delta = R2_full - R2_drop(pred)
  (computed using out-of-fold CV predictions)
- Create:
  (1) per-species barplots of predictor importance (summed/averaged across health metrics)
  (2) per-species heatmap-like alternative: a "lollipop matrix" (dot plot) of deltas
      (avoids another heatmap, but still compact)
  (3) one combined overview across species

Notes:
- This script does NOT do GeoGAM / Moran / spatial CV. It matches your "ndvi background investigations"
  species CSVs and produces GAM contribution summaries.
- If you want the spatial CV + Moran workflow across species too, we can extend this later.
"""

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from pygam import LinearGAM, s, l


# ----------------------------
# USER SETTINGS
# ----------------------------

# Each CSV should contain columns for: HEALTH_VARS + PREDICTORS + CONTROL_VARS
SPECIES_CSVS = {
    "Acer_platanoides": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer platanoides/ndvi_metrics_with_impervious.csv",
    "Acer_pseudoplatanus": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer pseudoplatanus/ndvi_metrics_with_impervious.csv",
    "Aesculus_hippocastanum": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/aesculus hippocastanum/ndvi_metrics_with_impervious.csv",
    "Platanus_x_acerifolia": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/platanus x acerifolia/ndvi_metrics_with_impervious.csv",
    "Tilia_x_euchlora": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/tilia x euchlora/ndvi_metrics_with_impervious.csv",
}

OUT_ROOT = Path("/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/_GAM_MULTI_SPECIES")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Variables
CONTROL_VARS = ["height"]

# Use a small, publishable subset (your “results section” set)
HEALTH_VARS = ["ndvi_peak", "sos_doy", "los_days", "amplitude", "auc_above_base_full"]

# Predictors you wanted in the small results matrix
# (You can add/remove here; this list drives the GAM comparisons)
PREDICTORS = [
    "imperv_10m",
    "poll_bc_anmean",
    "lst_temp_r50_y",
    "insolation9",
]

# Optional: if you want to still include other predictors as controls (besides height),
# put them here. They are always included as smooths but not ranked as "importance".
EXTRA_CONTROLS = []  # e.g. ["imperv_50m"]

# GAM settings
N_SPLINES = 10
LAM_GRID = np.logspace(-3, 3, 13)
DO_GRIDSEARCH = True

# CV settings
N_FOLDS = 5
RANDOM_STATE = 42

# Data cleaning / filtering
MIN_N = 80  # minimum rows per (health, model) to attempt

STANDARDIZE = True   # strongly recommended if you compare effect sizes across variables/species
EPS = 1e-12

BIOLOGICAL_FILTERS = {
    "los_days": lambda x: (x >= 60) & (x <= 365),
    "sos_doy": lambda x: (x >= 1) & (x <= 250),
    "ndvi_peak": lambda x: (x >= -0.1) & (x <= 1.0),
    "amplitude": lambda x: (x >= -0.1) & (x <= 1.0),
    "auc_above_base_full": lambda x: x > -1e9,  # keep broad; you can tighten if needed
    "height": lambda x: x > 1,
    "poll_bc_anmean": lambda x: x > 0,
}

# Plot settings
FIG_DPI = 240
POINT_ALPHA = 0.9


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def zscore(s: pd.Series, eps: float = 1e-12) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd < eps:
        return (s - mu) * 0.0
    return (s - mu) / sd

def make_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def apply_filters(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in BIOLOGICAL_FILTERS:
            out = out[BIOLOGICAL_FILTERS[c](out[c])]
    return out

def build_terms(n_cols: int, smooth_cols: list[int], linear_cols: list[int]) -> object:
    """
    Build pyGAM terms for a design matrix with n_cols features.
    smooth_cols: indices to use s()
    linear_cols: indices to use l()
    """
    terms = None
    for j in range(n_cols):
        if j in smooth_cols:
            t = s(j, n_splines=N_SPLINES)
        elif j in linear_cols:
            t = l(j)
        else:
            # default to smooth if unspecified (safe)
            t = s(j, n_splines=N_SPLINES)

        terms = t if terms is None else (terms + t)
    return terms

def fit_gam_cv_delta_r2(df_sub: pd.DataFrame, health: str, predictors: list[str], controls: list[str]) -> pd.DataFrame:
    """
    For one health metric, compute ΔR2 (explained deviance / pseudo-R2 proxy) per predictor:
      Δ = R2_full - R2_drop(pred)
    using K-fold CV predictions to reduce optimism.

    Returns a table with one row per predictor.
    """
    cols_full = predictors + controls + EXTRA_CONTROLS
    needed = [health] + cols_full
    d = df_sub[needed].dropna().copy()
    if len(d) < MIN_N:
        return pd.DataFrame()

    # standardize within this analysis subset (important!)
    if STANDARDIZE:
        for c in [health] + cols_full:
            d[c] = zscore(d[c], eps=EPS)

    y = d[health].values.astype(float)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Helper to CV-predict a given design matrix (X columns fixed)
    def cv_predict(X: np.ndarray) -> np.ndarray:
        yhat = np.full_like(y, np.nan, dtype=float)
        # smooth everything (controls included) – consistent with your setup
        smooth_cols = list(range(X.shape[1]))
        linear_cols = []  # keep it simple here
        terms = build_terms(X.shape[1], smooth_cols=smooth_cols, linear_cols=linear_cols)
        for tr, te in kf.split(X):
            gam = LinearGAM(terms)
            if DO_GRIDSEARCH:
                gam.gridsearch(X[tr], y[tr], lam=LAM_GRID, progress=False)
            else:
                gam.fit(X[tr], y[tr])
            yhat[te] = gam.predict(X[te])
        return yhat

    # FULL model
    X_full = d[cols_full].values.astype(float)
    yhat_full = cv_predict(X_full)
    r2_full = r2_score(y[np.isfinite(yhat_full)], yhat_full[np.isfinite(yhat_full)])

    rows = []
    for pred in predictors:
        cols_drop = [c for c in cols_full if c != pred]
        X_drop = d[cols_drop].values.astype(float)
        yhat_drop = cv_predict(X_drop)
        r2_drop = r2_score(y[np.isfinite(yhat_drop)], yhat_drop[np.isfinite(yhat_drop)])

        rows.append({
            "health_metric": health,
            "predictor": pred,
            "r2_full_cv": r2_full,
            "r2_drop_cv": r2_drop,
            "delta_r2": r2_full - r2_drop,
            "n": len(d),
        })

    return pd.DataFrame(rows)

def barplot_importance(df_imp: pd.DataFrame, title: str, outpath: Path):
    if df_imp.empty:
        return
    # aggregate across health metrics
    agg = (df_imp.groupby("predictor", as_index=False)
           .agg(delta_r2_mean=("delta_r2", "mean"),
                delta_r2_sum=("delta_r2", "sum")))
    # sort by mean
    agg = agg.sort_values("delta_r2_mean", ascending=False)

    plt.figure(figsize=(6.6, 3.8))
    plt.bar(agg["predictor"], agg["delta_r2_mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean ΔR² (CV)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()

def dotmatrix_importance(df_imp: pd.DataFrame, title: str, outpath: Path):
    """
    Heatmap alternative: dot-matrix (health metrics on y, predictors on x),
    dot size encodes |ΔR²| and x-position is fixed (categorical).
    """
    if df_imp.empty:
        return

    # keep order
    predictors = list(dict.fromkeys(df_imp["predictor"].tolist()))
    healths = list(dict.fromkeys(df_imp["health_metric"].tolist()))

    x_map = {p: i for i, p in enumerate(predictors)}
    y_map = {h: i for i, h in enumerate(healths)}

    xs = df_imp["predictor"].map(x_map).values
    ys = df_imp["health_metric"].map(y_map).values
    vals = df_imp["delta_r2"].values

    # scale point sizes
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    sizes = 40 + 700 * (np.abs(vals) / (vmax + 1e-12))

    plt.figure(figsize=(1.2 + 0.9 * len(predictors), 1.1 + 0.6 * len(healths)))
    ax = plt.gca()
    ax.scatter(xs, ys, s=sizes, alpha=POINT_ALPHA)

    ax.set_xticks(range(len(predictors)))
    ax.set_xticklabels(predictors, rotation=45, ha="right")
    ax.set_yticks(range(len(healths)))
    ax.set_yticklabels(healths)

    # annotate values (small)
    for x, y, v in zip(xs, ys, vals):
        ax.text(x, y, f"{v:.3f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Health metric")
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()

def combined_overview(df_all: pd.DataFrame, outpath: Path):
    """
    One combined barplot per species (small multiples) showing mean ΔR² per predictor.
    """
    if df_all.empty:
        return

    species_order = list(dict.fromkeys(df_all["species"].tolist()))
    preds = list(dict.fromkeys(df_all["predictor"].tolist()))

    nsp = len(species_order)
    fig_h = 2.2 + 1.6 * nsp
    plt.figure(figsize=(8.5, fig_h))
    ax = plt.gca()

    y0 = 0
    yticks = []
    ylabels = []

    for sp in species_order:
        sub = df_all[df_all["species"] == sp]
        agg = (sub.groupby("predictor", as_index=False)
               .agg(delta=("delta_r2", "mean")))
        # ensure all preds present
        agg = agg.set_index("predictor").reindex(preds).fillna(0.0).reset_index()

        # horizontal bars stacked by species blocks
        ys = y0 + np.arange(len(preds))
        ax.barh(ys, agg["delta"].values)
        yticks.extend(list(ys))
        ylabels.extend([f"{sp} · {p}" for p in agg["predictor"].tolist()])
        y0 = ys.max() + 2  # gap between species blocks

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Mean ΔR² (CV) when dropping predictor")
    ax.set_title("GAM predictor importance across species (non-spatial)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------

def main():
    all_rows = []

    for species, csv_path in SPECIES_CSVS.items():
        print(f"\n=== {species} ===")
        sp_out = OUT_ROOT / species
        ensure_dir(sp_out)

        df = pd.read_csv(csv_path)

        required = HEALTH_VARS + PREDICTORS + CONTROL_VARS + EXTRA_CONTROLS
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[skip] Missing columns in {species}: {missing}")
            continue

        df = make_numeric(df, required)

        # filters applied broadly (per-variable masks; will shrink df)
        df = apply_filters(df, required)

        species_results = []
        for health in HEALTH_VARS:
            df_imp = fit_gam_cv_delta_r2(df, health, PREDICTORS, CONTROL_VARS)
            if df_imp.empty:
                print(f"[warn] {species}: no result for {health} (n too small?)")
                continue
            df_imp["species"] = species
            species_results.append(df_imp)
            all_rows.append(df_imp)

        if not species_results:
            print(f"[skip] {species}: no health metric produced results")
            continue

        sp_res = pd.concat(species_results, ignore_index=True)
        sp_res.to_csv(sp_out / "gam_delta_r2_by_health_and_predictor.csv", index=False)

        # Per-species figures
        barplot_importance(
            sp_res,
            title=f"{species}: predictor importance (mean ΔR² across health metrics)",
            outpath=sp_out / "gam_importance_barplot.png"
        )
        dotmatrix_importance(
            sp_res,
            title=f"{species}: ΔR² by health metric (dot-matrix)",
            outpath=sp_out / "gam_importance_dotmatrix.png"
        )

        # Small "results table" subset (already small), but save a clean pivot too
        pivot = sp_res.pivot_table(index="health_metric", columns="predictor", values="delta_r2", aggfunc="mean")
        pivot.to_csv(sp_out / "gam_delta_r2_pivot.csv")

        print(f"[ok] saved: {sp_out}")

    # Combined overview across species
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_csv(OUT_ROOT / "ALL_species_gam_delta_r2_long.csv", index=False)
        combined_overview(all_df, OUT_ROOT / "ALL_species_gam_importance_overview.png")
        print(f"\n[ok] multi-species outputs in: {OUT_ROOT}")
    else:
        print("\n[warn] No species produced outputs (check missing columns / MIN_N).")


if __name__ == "__main__":
    main()
