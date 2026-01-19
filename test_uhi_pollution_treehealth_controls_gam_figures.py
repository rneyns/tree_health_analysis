#!/usr/bin/env python3
"""
Script: gam_dropone_importance_and_heatmaps.py

What this script adds vs your current one
----------------------------------------
For each HEALTH metric, it fits a *multi-predictor* GAM:

    health ~ s(height) + s(lst_temp_r50_y) + s(lst_temp_r100_y) + s(poll_no2) + s(poll_bc) + s(poll_pm10) + s(poll_pm25)

Then it estimates a defensible "importance-like" contribution for each predictor using
DROP-ONE DELTA explained deviance:

    ΔExplDev(pred j) = ExplDev(full) - ExplDev(reduced without pred j)

This is done per (health_metric, species CSV).

Outputs
-------
1) CSV tables:
   - gam_dropone_importance_long.csv
   - gam_dropone_importance_wide.csv
   - gam_dropone_threshold_flags.csv  (nonlinearity / threshold flags)

2) Figures:
   - heatmap_deltaExplDev_<health>.png              (species × predictor)
   - heatmap_deltaExplDev_<health>_with_flags.png   (threshold marked)
   - barplot_importance_<species>_<health>.png      (per species)
   - barplot_grid_<health>.png                      (all species in one big figure)

Threshold / nonlinearity indicator
---------------------------------
We mark a predictor as "nonlinear/threshold-like" if its smooth has:
    edof(term) >= THRESH_EDOF
(you can change the rule below)
"""

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from pygam import LinearGAM, s

# ----------------------------
# USER SETTINGS
# ----------------------------

# You said you have 5 CSVs (one per species). Put them here.
# Each CSV should contain columns for: HEALTH_VARS + PREDICTORS + CONTROL_VARS
SPECIES_CSVS = {
    "Acer_platanoides": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer platanoides/ndvi_metrics_with_impervious.csv",
    "Acer_pseudoplatanus": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer pseudoplatanus/ndvi_metrics_with_impervious.csv",
    "Aesculus_hippocastanum": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/aesculus hippocastanum/ndvi_metrics_with_impervious.csv",
    "Platanus_x_acerifolia": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/platanus x acerifolia/ndvi_metrics_with_impervious.csv",
    "Tilia_x_euchlora": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/tilia x euchlora/ndvi_metrics_with_impervious.csv",
}

OUT_DIR = Path("/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/gam_importance_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Controls (always included; not part of "importance" ranking by default)
CONTROL_VARS = ["height"]

# Predictors whose ΔExplDev you want (these will be in your heatmaps/barplots)
UHI_VARS = ["lst_temp_r50_y"]
POLLUTION_VARS = ["poll_no2_anmean", "poll_bc_anmean", "poll_pm10_anmean", "poll_pm25_anmean"]
PREDICTORS = UHI_VARS + POLLUTION_VARS

# Health metrics you want to model with multi-predictor GAMs
HEALTH_VARS = ["ndvi_peak", "sos_doy", "los_days", "amplitude", "auc_above_base_full"]

# GAM settings
N_SPLINES = 10
LAM_GRID = np.logspace(-3, 3, 15)

# Filter rows to biologically plausible values (same idea as your script)
BIOLOGICAL_FILTERS = {
    "los_days": lambda x: (x >= 100) & (x <= 365),
    "sos_doy": lambda x: (x >= 1) & (x <= 200),
    "peak_doy": lambda x: (x >= 100) & (x <= 300),
    "eos_doy": lambda x: (x >= 200) & (x <= 365),
    "amplitude": lambda x: (x > 0) & (x <= 1),
    "auc_above_base_full": lambda x: x > 0,
    "height": lambda x: x > 1,
    "poll_no2_anmean": lambda x: x > 0,
}

# Standardize within-species (recommended if you compare across predictors)
STANDARDIZE = True
STANDARDIZE_EPS = 1e-12

# Minimum n
MIN_N = 80

# Threshold flag rule (nonlinearity indicator)
THRESH_EDOF = 1.5

# Plot settings
FIG_DPI = 220
HEATMAP_VMAX = None  # set a number to clamp scale; None uses max
BAR_TOPK = None      # or set to an int, e.g. 6 to show top 6 predictors

# Predictor display names for plots
PRED_LABELS = {
    "lst_temp_r50_y": "LST (50 m)",
    "lst_temp_r100_y": "LST (100 m)",
    "poll_no2_anmean": "NO₂",
    "poll_bc_anmean": "BC",
    "poll_pm10_anmean": "PM₁₀",
    "poll_pm25_anmean": "PM₂.₅",
}

HEALTH_LABELS = {
    "ndvi_peak": "NDVI peak",
    "sos_doy": "Start of season (SOS)",
    "los_days": "Length of season (LOS)",
    "amplitude": "Amplitude",
    "auc_above_base_full": "AUC > base",
}


# ----------------------------
# HELPERS
# ----------------------------

def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def make_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def zscore_series(s: pd.Series, eps: float = 1e-12) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd < eps:
        return (s - mu) * 0.0
    return (s - mu) / sd

def apply_biological_filters(df: pd.DataFrame, vars_involved: list[str]) -> pd.DataFrame:
    out = df.copy()
    for v in vars_involved:
        if v in BIOLOGICAL_FILTERS:
            out = out[BIOLOGICAL_FILTERS[v](out[v])]
    return out

def build_terms(n_features: int):
    """Create s(0)+s(1)+...+s(n_features-1) with shared spline settings."""
    terms = s(0, n_splines=N_SPLINES)
    for j in range(1, n_features):
        terms = terms + s(j, n_splines=N_SPLINES)
    return terms

def fit_multigam(df_sub: pd.DataFrame, y_col: str, x_cols: list[str]):
    """
    Fit GAM with smooth term for each column in x_cols.
    Returns fitted gam + explained deviance + per-term edof (approx via gam.statistics_).
    """
    X = df_sub[x_cols].values.astype(float)
    y = df_sub[y_col].values.astype(float)

    gam = LinearGAM(build_terms(X.shape[1])).gridsearch(X, y, lam=LAM_GRID, progress=False)

    stats = gam.statistics_
    expl_dev = np.nan
    if "pseudo_r2" in stats and "explained_deviance" in stats["pseudo_r2"]:
        expl_dev = float(stats["pseudo_r2"]["explained_deviance"])

    # Per-term edof: pyGAM stores edof per coefficient and/or per term depending on version.
    # We'll attempt to extract per-term edof from statistics_. If unavailable, fallback to np.nan.
    term_edof = [np.nan] * X.shape[1]
    if "edof_per_term" in stats:
        try:
            ed = list(stats["edof_per_term"])
            if len(ed) >= X.shape[1]:
                term_edof = [float(v) for v in ed[:X.shape[1]]]
        except Exception:
            pass

    return gam, expl_dev, term_edof

def drop_one_importance(df_sub: pd.DataFrame, health: str, controls: list[str], predictors: list[str]):
    """
    Fits:
      full:  health ~ s(controls) + s(predictors)
      reduced: full minus one predictor each time
    Returns a list of dict rows with Δ explained deviance and threshold flags from EDoF.
    """
    x_full = controls + predictors
    gam_full, expl_full, edof_full = fit_multigam(df_sub, health, x_full)

    rows = []

    # Extract per-term EDoF for predictors from the FULL model (preferred for "shape" flags)
    # edof_full aligns with x_full column order
    edof_map = {col: edof_full[i] if i < len(edof_full) else np.nan for i, col in enumerate(x_full)}

    for pred in predictors:
        x_red = [c for c in x_full if c != pred]
        gam_red, expl_red, _ = fit_multigam(df_sub, health, x_red)

        delta = np.nan
        if np.isfinite(expl_full) and np.isfinite(expl_red):
            delta = expl_full - expl_red

        pred_edof = edof_map.get(pred, np.nan)
        threshold_flag = bool(np.isfinite(pred_edof) and pred_edof >= THRESH_EDOF)

        rows.append({
            "health_metric": health,
            "predictor": pred,
            "n": int(len(df_sub)),
            "explained_deviance_full": expl_full,
            "explained_deviance_reduced": expl_red,
            "delta_explained_deviance": delta,
            "edof_full_term": pred_edof,
            "threshold_flag_edof": threshold_flag,
        })

    return rows

def plot_heatmap_delta(mat: pd.DataFrame, flags: pd.DataFrame | None, title: str, outpath: Path):
    """
    mat: species × predictors (values = Δ explained deviance)
    flags: same shape boolean, True means "threshold/nonlinear" -> mark with '†'
    """
    A = mat.values.astype(float)

    vmax = HEATMAP_VMAX
    if vmax is None:
        vmax = np.nanmax(A) if np.isfinite(A).any() else 1.0
    vmax = float(max(vmax, 1e-9))

    fig_w = max(6.0, 1.2 + 1.0 * mat.shape[1])
    fig_h = max(3.5, 1.0 + 0.65 * mat.shape[0])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(A, vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_title(title, fontsize=12)

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([PRED_LABELS.get(c, c) for c in mat.columns], rotation=35, ha="right", fontsize=10)

    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=10)

    # annotate cells with value and optional flag symbol
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = A[i, j]
            if not np.isfinite(v):
                continue
            tag = ""
            if flags is not None:
                try:
                    if bool(flags.iloc[i, j]):
                        tag = "†"
                except Exception:
                    pass
            ax.text(j, i, f"{v:.02f}{tag}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Δ explained deviance (drop-one)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    '''
    # explain the symbol
    if flags is not None:
        ax.text(
            0.0, -0.12,
            f"† edof(term) ≥ {THRESH_EDOF} (nonlinear/threshold-like smooth)",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9
        )
    '''
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()

def plot_bar_per_species(mat: pd.DataFrame, flags: pd.DataFrame | None, health: str, outdir: Path):
    """
    Saves barplot per species for one health metric.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for species in mat.index:
        vals = mat.loc[species].copy()

        if BAR_TOPK is not None:
            vals = vals.sort_values(ascending=False).head(int(BAR_TOPK))
        else:
            vals = vals.sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
        ax.bar(range(len(vals)), vals.values)

        labels = [PRED_LABELS.get(c, c) for c in vals.index]
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=35, ha="right")

        # add dagger on xlabels if flagged
        if flags is not None:
            lbl2 = []
            for c in vals.index:
                dag = "†" if bool(flags.loc[species, c]) else ""
                lbl2.append(PRED_LABELS.get(c, c) + dag)
            ax.set_xticklabels(lbl2, rotation=35, ha="right")

        ax.set_ylabel("Δ explained deviance")
        ax.set_title(f"{species} — {HEALTH_LABELS.get(health, health)}", fontsize=12)
        plt.savefig(outdir / safe_filename(f"barplot_{species}_{health}.png"), dpi=FIG_DPI)
        plt.close()

def plot_bar_grid(mat: pd.DataFrame, flags: pd.DataFrame | None, health: str, outpath: Path):
    """
    One big multi-panel figure: barplot per species for one health metric.
    """
    n_species = mat.shape[0]
    ncols = 2
    nrows = int(np.ceil(n_species / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for k, species in enumerate(mat.index):
        ax = axes[k]
        vals = mat.loc[species].copy().sort_values(ascending=False)
        if BAR_TOPK is not None:
            vals = vals.head(int(BAR_TOPK))

        ax.bar(range(len(vals)), vals.values)
        labels = [PRED_LABELS.get(c, c) for c in vals.index]

        if flags is not None:
            labels = [lab + ("†" if bool(flags.loc[species, c]) else "") for lab, c in zip(labels, vals.index)]

        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_title(species, fontsize=11)
        ax.set_ylabel("ΔExplDev")

    # hide unused panels
    for j in range(k + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Drop-one Δ explained deviance — {HEALTH_LABELS.get(health, health)}", fontsize=14)
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------

def main():
    out_tables = OUT_DIR / "tables"
    out_figs = OUT_DIR / "figures"
    out_tables.mkdir(exist_ok=True)
    out_figs.mkdir(exist_ok=True)

    long_rows = []

    for species, csv_path in SPECIES_CSVS.items():
        df = pd.read_csv(csv_path)

        needed = list(dict.fromkeys(HEALTH_VARS + CONTROL_VARS + PREDICTORS))
        df = make_numeric(df, needed)

        for health in HEALTH_VARS:
            cols_needed = [health] + CONTROL_VARS + PREDICTORS
            df_sub = df[cols_needed].dropna().copy()

            df_sub = apply_biological_filters(df_sub, cols_needed)

            n = len(df_sub)
            if n < MIN_N:
                print(f"[skip] {species} {health}: n={n} < {MIN_N}")
                continue

            # standardize inside subset (recommended for comparability)
            if STANDARDIZE:
                for c in cols_needed:
                    df_sub[c] = zscore_series(df_sub[c], eps=STANDARDIZE_EPS)

            # compute drop-one contributions for predictors
            rows = drop_one_importance(df_sub, health, CONTROL_VARS, PREDICTORS)
            for r in rows:
                r["species"] = species
            long_rows.extend(rows)

            print(f"[ok] {species} {health}: n={n}")

    res_long = pd.DataFrame(long_rows)
    if res_long.empty:
        raise RuntimeError("No results computed. Check CSV paths, column names, and MIN_N.")

    # Save long table
    res_long.to_csv(out_tables / "gam_dropone_importance_long.csv", index=False)

    # Wide table: species × predictor per health
    # We'll also keep threshold flags separately
    for health in sorted(res_long["health_metric"].unique()):
        df_h = res_long[res_long["health_metric"] == health].copy()

        mat = df_h.pivot_table(index="species", columns="predictor", values="delta_explained_deviance", aggfunc="mean")
        mat = mat.reindex(columns=PREDICTORS)

        flags = df_h.pivot_table(index="species", columns="predictor", values="threshold_flag_edof", aggfunc="max")
        flags = flags.reindex(columns=PREDICTORS)

        # Save matrices
        mat.to_csv(out_tables / f"deltaExplDev_matrix_{health}.csv")
        flags.to_csv(out_tables / f"threshold_flags_matrix_{health}.csv")

        # Heatmaps
        plot_heatmap_delta(
            mat,
            flags,
            title=f"Δ explained deviance (drop-one) — {HEALTH_LABELS.get(health, health)}",
            outpath=out_figs / f"heatmap_deltaExplDev_{health}.png"
        )

        # Bar plots: per species and grid
        bars_dir = out_figs / f"barplots_{health}"
        plot_bar_per_species(mat, flags, health, bars_dir)
        plot_bar_grid(mat, flags, health, out_figs / f"barplot_grid_{health}.png")

    # Also produce one combined wide export for convenience
    # (stack health into a MultiIndex columns)
    mats = []
    for health in sorted(res_long["health_metric"].unique()):
        df_h = res_long[res_long["health_metric"] == health].copy()
        mat = df_h.pivot_table(index="species", columns="predictor", values="delta_explained_deviance", aggfunc="mean")
        mat = mat.reindex(columns=PREDICTORS)
        mat.columns = pd.MultiIndex.from_product([[health], mat.columns])
        mats.append(mat)

    wide_all = pd.concat(mats, axis=1)
    wide_all.to_csv(out_tables / "gam_dropone_importance_wide.csv")

    print("\n✔ Done!")
    print(f"Tables: {out_tables.resolve()}")
    print(f"Figures: {out_figs.resolve()}")
    print("Note: † marks edof(term) >= THRESH_EDOF (nonlinear/threshold-like smooth).")


if __name__ == "__main__":
    main()
