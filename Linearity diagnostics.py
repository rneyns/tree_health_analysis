#!/usr/bin/env python3
"""
Linearity diagnostics supplement for the multi-species MLR workflow.

Adds two linearity checks to the existing OLS pipeline:
  1. RESET test (Ramsey 1969)      — formal test for functional form misspecification
  2. Spearman vs Pearson comparison — flags monotonic non-linearity per predictor

These are run for every species × health metric combination and produce:

PER SPECIES × HEALTH METRIC (saved in same folder structure as main script)
  {health}_RESET_test.txt          — RESET test results (F-stat, p-value, interpretation)
  {health}_linearity_scatter.png   — scatterplots with Pearson r, Spearman ρ, and Δr

COMBINED ACROSS ALL SPECIES
  ALL_RESET_summary.csv            — one row per species × health metric
  ALL_RESET_heatmap.png            — heatmap of RESET p-values (red = misspecified)
  ALL_pearson_spearman_delta.png   — heatmap of |Pearson r − Spearman ρ| per
                                     predictor × health metric (averaged across species)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESET test:
  H0: the linear functional form is correctly specified.
  Rejection (p < 0.05): adding squared/cubed fitted values significantly
  improves the model → evidence of non-linearity or omitted variables.
  Note: the RESET test is sensitive to any misspecification, not only
  non-linearity. A significant result warrants visual inspection of
  added-variable plots before concluding non-linearity is the cause.

Spearman vs Pearson:
  |ρ_Spearman − r_Pearson| > 0.10 suggests the relationship is better
  described by a monotonic function than a linear one — i.e. the ranks
  are more regularly related than the raw values.
  Large discrepancies combined with a significant RESET test provide
  converging evidence of non-linearity for that predictor.

References:
  Ramsey, J. B. (1969). Tests for specification errors in classical
      linear least squares regression analysis. Journal of the Royal
      Statistical Society Series B, 31(2), 350–371.
  Legendre, P. & Legendre, L. (2012). Numerical Ecology, 3rd ed.
      Elsevier. [Spearman correlation, Chapter 5]
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd

# ── Copy your settings from the main script exactly ──────────────────────────

BASE_FONT  = 12
TITLE_FONT = 13

mpl.rcParams.update({
    "font.size":        BASE_FONT,
    "axes.titlesize":   TITLE_FONT,
    "axes.labelsize":   BASE_FONT,
    "xtick.labelsize":  BASE_FONT - 1,
    "ytick.labelsize":  BASE_FONT - 1,
    "legend.fontsize":  BASE_FONT - 1,
    "figure.titlesize": TITLE_FONT,
})

SPECIES_CSVS = {
    "Acer_platanoides":       "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer platanoides/ndvi_metrics_with_impervious.csv",
    "Acer_pseudoplatanus":    "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer pseudoplatanus/ndvi_metrics_with_impervious.csv",
    "Aesculus_hippocastanum": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/aesculus hippocastanum/ndvi_metrics_with_impervious.csv",
    "Platanus_x_acerifolia":  "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/platanus x acerifolia/ndvi_metrics_with_impervious.csv",
    "Tilia_x_euchlora":       "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/tilia x euchlora/ndvi_metrics_with_impervious.csv",
}

SPECIES_LABELS = {
    "Acer_platanoides":       "Acer platanoides",
    "Acer_pseudoplatanus":    "Acer pseudoplatanus",
    "Aesculus_hippocastanum": "Aesculus hippocastanum",
    "Platanus_x_acerifolia":  "Platanus × acerifolia",
    "Tilia_x_euchlora":       "Tilia × euchlora",
}

HEALTH_LABELS = {
    "ndvi_peak":           "Peak NDVI",
    "sos_doy":             "Start of season (DOY)",
    "los_days":            "Length of season (days)",
    "amplitude":           "NDVI amplitude",
    "auc_above_base_full": "Seasonal NDVI integral",
}

PREDICTOR_LABELS = {
    "imperv_100m":     "Impervious surface (100 m)",
    "poll_bc_anmean":  "Black carbon (annual mean)",
    "lst_temp_r100_y": "Land surface temperature (100 m)",
    "insolation9":     "Solar radiation",
}

OUT_ROOT = Path(
    "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/"
    "Data analysis/ndvi background investigations/_MLR_SPATIAL_v1"
)

SHAPEFILE_PATH             = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/crown_shapes_final_CRS.shp"
CSV_ID_COL                 = "tree_id"
SHP_ID_COL                 = "crown_id"
USE_CENTROID_IF_NOT_POINTS = True
TARGET_CRS                 = None

CONTROL_VARS = ["height"]
HEALTH_VARS  = ["ndvi_peak", "sos_doy", "los_days", "amplitude", "auc_above_base_full"]
PREDICTORS   = ["imperv_100m", "poll_bc_anmean", "lst_temp_r100_y", "insolation9"]
COORD_COLS   = ["x", "y"]

MIN_N   = 30
ALPHA   = 0.05
FIG_DPI = 200

# Threshold for flagging Spearman–Pearson discrepancy as noteworthy
DELTA_THRESHOLD = 0.10

BIOLOGICAL_FILTERS = {
    "los_days":            lambda x: (x >= 60)   & (x <= 365),
    "sos_doy":             lambda x: (x >= 1)    & (x <= 250),
    "ndvi_peak":           lambda x: (x >= -0.1) & (x <= 1.0),
    "amplitude":           lambda x: (x >= -0.1) & (x <= 1.0),
    "auc_above_base_full": lambda x: x > -1e9,
    "height":              lambda x: x > 1,
    "poll_bc_anmean":      lambda x: x > 0,
}


# ── Utility (copied from main script) ────────────────────────────────────────

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def make_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def apply_filters(df, cols):
    out = df.copy()
    for c in cols:
        if c in BIOLOGICAL_FILTERS and c in out.columns:
            out = out[BIOLOGICAL_FILTERS[c](out[c])]
    return out

def _normalize_id_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    frac = np.abs(x - np.round(x))
    x = x.where((~x.isna()) & (frac < 1e-9), x)
    return np.round(x).astype("Int64")

def _normalize_id_str(s):
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"\s+", "", regex=True)
    return out.str.lower().replace("nan", pd.NA)

def attach_xy_from_shapefile(df):
    if all(c in df.columns for c in COORD_COLS):
        return df
    if not SHAPEFILE_PATH:
        raise ValueError("SHAPEFILE_PATH not set.")
    if CSV_ID_COL not in df.columns:
        raise ValueError(f"Missing join key '{CSV_ID_COL}'.")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    geom = gdf.geometry.centroid if USE_CENTROID_IF_NOT_POINTS else gdf.geometry
    xn, yn = COORD_COLS
    gdf_xy = gdf[[SHP_ID_COL]].copy()
    gdf_xy[xn] = geom.x
    gdf_xy[yn] = geom.y
    df2 = df.copy()
    df2["_jn"]    = _normalize_id_numeric(df2[CSV_ID_COL])
    gdf_xy["_jn"] = _normalize_id_numeric(gdf_xy[SHP_ID_COL])
    merged = df2.merge(gdf_xy[["_jn", xn, yn]], how="left", on="_jn").drop(columns=["_jn"])
    if merged[[xn, yn]].notna().all(axis=1).sum() == 0:
        df2["_js"]    = _normalize_id_str(df2[CSV_ID_COL])
        gdf_xy["_js"] = _normalize_id_str(gdf_xy[SHP_ID_COL])
        merged = df2.merge(gdf_xy[["_js", xn, yn]], how="left", on="_js"
                           ).drop(columns=["_jn", "_js"], errors="ignore")
    return merged


# ── RESET test ────────────────────────────────────────────────────────────────

def run_reset_test(y: np.ndarray, X_sm: pd.DataFrame) -> dict:
    """
    Ramsey RESET test for functional form misspecification.

    Tests H0: the linear model is correctly specified, by checking whether
    powers (squared and cubed) of the fitted values add significant
    explanatory power.

    Uses statsmodels linear_reset with power=2 (adds ŷ² and ŷ³)
    and an F-test for joint significance.

    Returns a dict with F-statistic, p-value, degrees of freedom,
    and a plain-English interpretation.
    """
    try:
        ols  = sm.OLS(y, X_sm).fit()
        test = linear_reset(ols, power=2, use_f=True)
        f_stat = float(test.fvalue)
        p_val  = float(test.pvalue)
        df_num = int(test.df_num)
        df_den = int(test.df_denom)

        if p_val < 0.001:
            strength = "strong evidence of misspecification"
        elif p_val < 0.01:
            strength = "moderate evidence of misspecification"
        elif p_val < 0.05:
            strength = "mild evidence of misspecification"
        else:
            strength = "no significant evidence of misspecification"

        return {
            "f_stat":        f_stat,
            "p_value":       p_val,
            "df_num":        df_num,
            "df_den":        df_den,
            "significant":   p_val < ALPHA,
            "interpretation": strength,
            "r2_ols":        float(ols.rsquared),
            "r2_adj_ols":    float(ols.rsquared_adj),
            "n":             len(y),
        }
    except Exception as e:
        return {
            "f_stat": np.nan, "p_value": np.nan,
            "df_num": np.nan, "df_den": np.nan,
            "significant": False,
            "interpretation": f"RESET test failed: {e}",
            "r2_ols": np.nan, "r2_adj_ols": np.nan, "n": len(y),
        }


# ── Spearman vs Pearson ───────────────────────────────────────────────────────

def spearman_pearson_comparison(d: pd.DataFrame,
                                 health: str,
                                 predictors: list) -> pd.DataFrame:
    """
    For each predictor, compute:
      - Pearson r   (linear association)
      - Spearman ρ  (monotonic association)
      - |Δ| = |ρ − r|  (discrepancy suggesting non-linearity)
      - p-values for both

    A large |Δ| means ranks correlate more regularly than raw values,
    suggesting a monotonic but non-linear relationship.
    """
    rows = []
    y = d[health].values
    for pred in predictors:
        x = d[pred].values
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]
        if len(xm) < 5:
            continue

        r_p,  p_p  = stats.pearsonr(xm, ym)
        r_sp, p_sp = stats.spearmanr(xm, ym)
        delta = abs(r_sp - r_p)

        rows.append({
            "predictor":      pred,
            "predictor_label": PREDICTOR_LABELS.get(pred, pred),
            "pearson_r":      round(r_p,  4),
            "pearson_p":      round(p_p,  4),
            "spearman_rho":   round(r_sp, 4),
            "spearman_p":     round(p_sp, 4),
            "abs_delta":      round(delta, 4),
            "flagged":        delta > DELTA_THRESHOLD,
            "n":              int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ── Per-metric plots ──────────────────────────────────────────────────────────

def plot_linearity_scatter(d: pd.DataFrame,
                            health: str,
                            predictors: list,
                            sp_pretty: str,
                            sp_df: pd.DataFrame,
                            outpath: Path) -> None:
    """
    Scatterplot grid: one panel per predictor.
    Each panel shows raw data, OLS regression line, and annotates:
      - Pearson r (p-value)
      - Spearman ρ (p-value)
      - |Δ| flagged if > DELTA_THRESHOLD
    This provides the visual complement to the RESET test.
    """
    h_lbl = HEALTH_LABELS.get(health, health)
    ncols = 2
    nrows = int(np.ceil(len(predictors) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6.5 * ncols, 5 * nrows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, pred in enumerate(predictors):
        ax    = axes[i]
        p_lbl = PREDICTOR_LABELS.get(pred, pred)

        x = d[pred].values
        y = d[health].values
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]

        ax.scatter(xm, ym, s=14, alpha=0.30, color="steelblue",
                   linewidths=0, zorder=2)

        if len(xm) > 2:
            # OLS fit line
            Xc   = sm.add_constant(xm, has_constant="add")
            fit  = sm.OLS(ym, Xc).fit()
            xr   = np.linspace(xm.min(), xm.max(), 300)
            xr_c = sm.add_constant(xr, has_constant="add")
            pred_ci = fit.get_prediction(xr_c).summary_frame(alpha=ALPHA)

            ax.plot(xr, pred_ci["mean"], color="firebrick", lw=1.8, zorder=3)
            ax.fill_between(xr,
                            pred_ci["mean_ci_lower"],
                            pred_ci["mean_ci_upper"],
                            color="firebrick", alpha=0.12, zorder=1)

            # Correlation stats
            r_p,  p_p  = stats.pearsonr(xm, ym)
            r_sp, p_sp = stats.spearmanr(xm, ym)
            delta      = abs(r_sp - r_p)
            flag       = "⚠ |Δ| large" if delta > DELTA_THRESHOLD else ""

            def _sig(p):
                return ("***" if p < 0.001 else "**" if p < 0.01
                        else "*" if p < 0.05 else "ns")

            ann = (f"Pearson r  = {r_p:+.3f} {_sig(p_p)}\n"
                   f"Spearman ρ = {r_sp:+.3f} {_sig(p_sp)}\n"
                   f"|Δ| = {delta:.3f}  {flag}")

            box_color = "#fff3cd" if delta > DELTA_THRESHOLD else "white"
            ax.text(0.03, 0.97, ann,
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=BASE_FONT - 3,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=box_color, alpha=0.85,
                              edgecolor="lightgrey"))

        ax.set_xlabel(p_lbl, fontsize=BASE_FONT)
        ax.set_ylabel(h_lbl, fontsize=BASE_FONT)
        ax.set_title(p_lbl, fontsize=TITLE_FONT)

    for j in range(len(predictors), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{sp_pretty} – {h_lbl}\n"
        f"Pearson r vs Spearman ρ per predictor\n"
        f"Yellow box: |Δ| > {DELTA_THRESHOLD} (possible non-linearity)",
        fontsize=TITLE_FONT
    )
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


# ── Combined summary plots ────────────────────────────────────────────────────

def plot_reset_heatmap(reset_df: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap of RESET test p-values across species × health metrics.
    Red cells = significant misspecification (p < 0.05).
    Annotated with p-value and F-statistic.
    """
    species_list = list(reset_df["species"].unique())
    health_list  = [HEALTH_LABELS.get(h, h) for h in HEALTH_VARS
                    if HEALTH_LABELS.get(h, h) in reset_df["health_metric"].values]

    # Build matrix of p-values
    mat_p = np.full((len(species_list), len(health_list)), np.nan)
    mat_f = np.full((len(species_list), len(health_list)), np.nan)

    for _, row in reset_df.iterrows():
        si = species_list.index(row["species"]) if row["species"] in species_list else -1
        hi = health_list.index(row["health_metric"]) if row["health_metric"] in health_list else -1
        if si >= 0 and hi >= 0:
            mat_p[si, hi] = row["reset_p"]
            mat_f[si, hi] = row["reset_f"]

    # Colour by -log10(p) so smaller p = darker red
    with np.errstate(divide="ignore"):
        mat_log = -np.log10(np.clip(mat_p, 1e-10, 1.0))

    fig, ax = plt.subplots(figsize=(2.4 * len(health_list),
                                     0.8 + 0.7 * len(species_list)))
    im = ax.imshow(mat_log, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=max(3.0, np.nanmax(mat_log)))

    for i in range(len(species_list)):
        for j in range(len(health_list)):
            p = mat_p[i, j]
            f = mat_f[i, j]
            if np.isfinite(p):
                sig = "*" if p < ALPHA else ""
                txt = f"p={p:.3f}{sig}\nF={f:.1f}"
                dark = mat_log[i, j] > np.nanmax(mat_log) * 0.6
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=BASE_FONT - 3,
                        color="white" if dark else "black")

    ax.set_xticks(range(len(health_list)))
    ax.set_xticklabels(health_list, rotation=35, ha="right", fontsize=BASE_FONT - 1)
    ax.set_yticks(range(len(species_list)))
    ax.set_yticklabels(species_list, fontsize=BASE_FONT - 1)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("−log₁₀(p)  [darker = stronger evidence of misspecification]",
                   fontsize=BASE_FONT)

    ax.set_title(
        "RESET test for functional form misspecification\n"
        f"* = p < {ALPHA}  |  Darker = stronger evidence of non-linearity",
        fontsize=TITLE_FONT
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_delta_heatmap(sp_df: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap of mean |Pearson r − Spearman ρ| per predictor × health metric,
    averaged across species.
    Larger values suggest stronger monotonic non-linearity for that
    predictor-outcome combination.
    """
    pred_codes   = PREDICTORS
    pred_labels  = [PREDICTOR_LABELS.get(p, p) for p in pred_codes]
    health_list  = [HEALTH_LABELS.get(h, h) for h in HEALTH_VARS]

    agg = (sp_df.groupby(["health_metric", "predictor"])["abs_delta"]
           .mean().reset_index())

    mat = np.full((len(pred_codes), len(health_list)), np.nan)
    for _, row in agg.iterrows():
        pi = pred_codes.index(row["predictor"]) if row["predictor"] in pred_codes else -1
        hi = health_list.index(row["health_metric"]) if row["health_metric"] in health_list else -1
        if pi >= 0 and hi >= 0:
            mat[pi, hi] = row["abs_delta"]

    fig, ax = plt.subplots(figsize=(2.4 * len(health_list),
                                     0.8 + 0.7 * len(pred_codes)))
    vmax = max(np.nanmax(mat), DELTA_THRESHOLD + 0.01) if np.any(np.isfinite(mat)) else 0.3
    im   = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

    for i in range(len(pred_codes)):
        for j in range(len(health_list)):
            v = mat[i, j]
            if np.isfinite(v):
                flag = " ⚠" if v > DELTA_THRESHOLD else ""
                dark = v > vmax * 0.6
                ax.text(j, i, f"{v:.3f}{flag}",
                        ha="center", va="center", fontsize=BASE_FONT - 2,
                        color="white" if dark else "black")

    ax.set_xticks(range(len(health_list)))
    ax.set_xticklabels(health_list, rotation=35, ha="right", fontsize=BASE_FONT - 1)
    ax.set_yticks(range(len(pred_codes)))
    ax.set_yticklabels(pred_labels, fontsize=BASE_FONT - 1)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"|Pearson r − Spearman ρ|  (mean across species)", fontsize=BASE_FONT)

    ax.set_title(
        f"|Pearson r − Spearman ρ| per predictor × health metric\n"
        f"⚠ = mean |Δ| > {DELTA_THRESHOLD}  (possible monotonic non-linearity)",
        fontsize=TITLE_FONT
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


# ── Per-metric core function ──────────────────────────────────────────────────

def run_linearity_diagnostics(df_sub: pd.DataFrame,
                               health: str,
                               predictors: list,
                               controls: list,
                               sp_pretty: str,
                               out_dir: Path) -> tuple:
    """
    Run RESET test and Spearman–Pearson comparison for one
    species × health metric combination.

    Returns (reset_row dict, sp_rows list of dicts).
    """
    all_x_cols = predictors + controls
    needed     = [health] + all_x_cols

    d = df_sub[needed].dropna().copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    if len(d) < MIN_N:
        print(f"    [skip] {health}: n={len(d)} < {MIN_N}")
        return None, []

    h_safe = health.replace("/", "_")
    h_lbl  = HEALTH_LABELS.get(health, health)

    y    = d[health].values.astype(float)
    X_df = d[all_x_cols].copy()
    X_sm = sm.add_constant(X_df, has_constant="add")

    # ── RESET test ────────────────────────────────────────────
    reset = run_reset_test(y, X_sm)

    # Write text result
    with open(out_dir / f"{h_safe}_RESET_test.txt", "w") as f:
        f.write(f"RESET test — {sp_pretty} | {h_lbl}\n")
        f.write(f"{'='*55}\n")
        f.write(f"  F({reset['df_num']}, {reset['df_den']}) = {reset['f_stat']:.4f}\n")
        f.write(f"  p-value = {reset['p_value']:.4f}\n")
        f.write(f"  OLS R²  = {reset['r2_ols']:.4f}  (adj. R² = {reset['r2_adj_ols']:.4f})\n")
        f.write(f"  n = {reset['n']}\n\n")
        f.write(f"  Interpretation: {reset['interpretation']}\n\n")
        f.write(
            "  Note: the RESET test detects any functional form misspecification,\n"
            "  not only non-linearity. A significant result should be followed up\n"
            "  with visual inspection of added-variable (partial regression) plots.\n"
        )

    # ── Spearman vs Pearson ───────────────────────────────────
    sp_df = spearman_pearson_comparison(d, health, predictors)
    sp_df["species"]       = sp_pretty
    sp_df["health_metric"] = h_lbl

    # ── Scatterplot with both correlations ────────────────────
    plot_linearity_scatter(
        d         = d,
        health    = health,
        predictors = predictors,
        sp_pretty  = sp_pretty,
        sp_df      = sp_df,
        outpath    = out_dir / f"{h_safe}_linearity_scatter.png",
    )

    reset_row = {
        "species":       sp_pretty,
        "health_metric": h_lbl,
        "reset_f":       reset["f_stat"],
        "reset_p":       reset["p_value"],
        "reset_df_num":  reset["df_num"],
        "reset_df_den":  reset["df_den"],
        "reset_sig":     reset["significant"],
        "reset_interp":  reset["interpretation"],
        "r2_ols":        reset["r2_ols"],
        "r2_adj_ols":    reset["r2_adj_ols"],
        "n":             reset["n"],
    }

    return reset_row, sp_df.to_dict("records")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_reset = []
    all_sp    = []   # Spearman–Pearson rows

    for species, csv_path in SPECIES_CSVS.items():
        print(f"\n{'='*55}\n  {species}\n{'='*55}")
        sp_out    = OUT_ROOT / species
        ensure_dir(sp_out)
        sp_pretty = SPECIES_LABELS.get(species, species)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [skip] Cannot read CSV: {e}")
            continue

        required = HEALTH_VARS + PREDICTORS + CONTROL_VARS
        missing  = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [skip] Missing columns: {missing}")
            continue

        try:
            df = attach_xy_from_shapefile(df)
        except Exception as e:
            print(f"  [skip] Coordinate join failed: {e}")
            continue

        df = make_numeric(df, required + COORD_COLS)
        df = apply_filters(df, required + COORD_COLS)
        print(f"  n = {len(df)} trees after filtering")

        for health in HEALTH_VARS:
            print(f"  → {HEALTH_LABELS.get(health, health)}")
            h_out = sp_out / health
            ensure_dir(h_out)

            reset_row, sp_rows = run_linearity_diagnostics(
                df_sub     = df,
                health     = health,
                predictors = PREDICTORS,
                controls   = CONTROL_VARS,
                sp_pretty  = sp_pretty,
                out_dir    = h_out,
            )
            if reset_row:
                all_reset.append(reset_row)
            all_sp.extend(sp_rows)

        print(f"  [ok] → {sp_out}")

    # ── Combined outputs ──────────────────────────────────────
    if all_reset:
        reset_df = pd.DataFrame(all_reset)
        reset_df.to_csv(OUT_ROOT / "ALL_RESET_summary.csv", index=False)
        plot_reset_heatmap(reset_df, OUT_ROOT / "ALL_RESET_heatmap.png")
        print(f"\nRESET summary: {reset_df['reset_sig'].sum()} / {len(reset_df)} "
              f"species × metric combinations show significant misspecification.")

    if all_sp:
        sp_df = pd.DataFrame(all_sp)
        sp_df.to_csv(OUT_ROOT / "ALL_pearson_spearman.csv", index=False)
        plot_delta_heatmap(sp_df, OUT_ROOT / "ALL_pearson_spearman_delta.png")
        n_flagged = sp_df["flagged"].sum()
        print(f"Spearman–Pearson: {n_flagged} / {len(sp_df)} "
              f"predictor × metric × species combinations flagged (|Δ| > {DELTA_THRESHOLD}).")

    print(f"\n[ok] All linearity diagnostics saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()