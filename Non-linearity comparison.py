#!/usr/bin/env python3
"""
Non-linearity comparison for peak NDVI (and other health metrics).

Fits four functional forms for each predictor × health metric × species:
  1. Linear          : y ~ X  (baseline OLS)
  2. Log-transformed : y ~ log(X)  (logarithmic decay/growth)
  3. Quadratic       : y ~ X + X²  (saturating or hump-shaped)
  4. GAM             : y ~ s(X)  (data-driven smooth, via pyGAM)

Model comparison is based on AIC. The best-fitting functional form
per predictor is identified and reported.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear vs log:
  If AIC_log < AIC_linear by > 2: the relationship is better described
  by a logarithmic function — consistent with a saturating effect
  (e.g. NDVI drops steeply at low imperviousness then flattens).

Quadratic:
  If the squared term is significant and AIC improves: the relationship
  is hump-shaped or U-shaped (e.g. intermediate imperviousness is
  most/least stressful).

GAM smooth:
  The most flexible form. The smooth plot reveals the actual shape
  without imposing any functional form. If the GAM smooth is
  approximately linear: the linear model is adequate. If it curves:
  the GAM partial effect plot shows exactly how.

ΔAIC > 2   : meaningful improvement
ΔAIC > 10  : strong improvement
ΔAIC > 100 : overwhelming improvement (as seen for peak NDVI)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PER SPECIES × HEALTH METRIC
  {health}_functional_form_comparison.png  — 4-panel: raw scatter with
      linear/log/quadratic fits overlaid + GAM smooth per predictor
  {health}_functional_form_AIC.txt         — AIC table + best model

COMBINED
  ALL_functional_form_AIC.csv              — full AIC comparison table
  ALL_functional_form_winner.png           — heatmap of winning functional
      form per predictor × health metric × species
  ALL_log_vs_linear_DAIC.png              — heatmap of ΔAIC(log−linear)
      per predictor × health metric (averaged across species); negative
      = log wins

Dependencies:
  pip install numpy pandas matplotlib scipy statsmodels geopandas pygam
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.api as sm
import geopandas as gpd

# Optional GAM — gracefully skip if not installed
try:
    from pygam import LinearGAM, s, l as gam_l
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("[warn] pyGAM not installed. GAM fits will be skipped.\n"
          "       Install with: pip install pygam")

# ── Settings (copy from your main script) ────────────────────────────────────

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
    "Data analysis/ndvi background investigations/nonlinearity_comparison"
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

# Colours for the four functional forms
FORM_COLOURS = {
    "linear":    "#4C72B0",
    "log":       "#DD8452",
    "quadratic": "#55A868",
    "gam":       "#C44E52",
}

BIOLOGICAL_FILTERS = {
    "los_days":            lambda x: (x >= 60)   & (x <= 365),
    "sos_doy":             lambda x: (x >= 1)    & (x <= 250),
    "ndvi_peak":           lambda x: (x >= -0.1) & (x <= 1.0),
    "amplitude":           lambda x: (x >= -0.1) & (x <= 1.0),
    "auc_above_base_full": lambda x: x > -1e9,
    "height":              lambda x: x > 1,
    "poll_bc_anmean":      lambda x: x > 0,
}


# ── Utility ───────────────────────────────────────────────────────────────────

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


# ── Model fitting helpers ─────────────────────────────────────────────────────

def _aic_ols(ols_result) -> float:
    return float(ols_result.aic)


def fit_univariate_forms(x: np.ndarray,
                          y: np.ndarray,
                          controls: np.ndarray) -> dict:
    """
    Fit four functional forms for the bivariate x → y relationship,
    controlling for the control variables (height) in all models.

    The controls matrix is included in every model to ensure AIC
    comparisons are fair (same outcome, same controls, different
    functional form for the focal predictor only).

    Returns a dict with keys: linear, log, quadratic, gam.
    Each value is a dict with: aic, r2, r2_adj, coef, fitted, x_grid, y_grid.
    x_grid / y_grid are for plotting the smooth curve.
    """
    n = len(y)
    # Build base control matrix (intercept + controls)
    if controls.shape[1] > 0:
        X_ctrl = sm.add_constant(controls, has_constant="add")
    else:
        X_ctrl = np.ones((n, 1))

    results = {}

    # ── 1. Linear ─────────────────────────────────────────────
    X_lin = np.column_stack([X_ctrl, x])
    try:
        ols_lin = sm.OLS(y, X_lin).fit()
        x_grid  = np.linspace(x.min(), x.max(), 300)
        # Predict at mean controls
        ctrl_mean = np.full((300, X_ctrl.shape[1]), X_ctrl.mean(axis=0))
        X_pred    = np.column_stack([ctrl_mean, x_grid])
        y_grid    = X_pred @ ols_lin.params
        results["linear"] = {
            "aic":    _aic_ols(ols_lin),
            "r2":     ols_lin.rsquared,
            "r2_adj": ols_lin.rsquared_adj,
            "coef":   float(ols_lin.params[-1]),
            "se":     float(ols_lin.bse[-1]),
            "p":      float(ols_lin.pvalues[-1]),
            "fitted": ols_lin.fittedvalues,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "label":  "Linear",
        }
    except Exception as e:
        results["linear"] = {"aic": np.nan, "label": "Linear", "error": str(e)}

    # ── 2. Log-transformed ────────────────────────────────────
    # Requires x > 0; shift by small epsilon if needed
    x_pos = x - x.min() + 1e-6 if x.min() <= 0 else x
    x_log = np.log(x_pos)
    X_log = np.column_stack([X_ctrl, x_log])
    try:
        ols_log = sm.OLS(y, X_log).fit()
        x_grid_pos = np.linspace(x_pos.min(), x_pos.max(), 300)
        x_grid     = x_grid_pos - 1e-6 + x.min() if x.min() <= 0 else x_grid_pos
        ctrl_mean  = np.full((300, X_ctrl.shape[1]), X_ctrl.mean(axis=0))
        X_pred     = np.column_stack([ctrl_mean, np.log(x_grid_pos)])
        y_grid     = X_pred @ ols_log.params
        results["log"] = {
            "aic":    _aic_ols(ols_log),
            "r2":     ols_log.rsquared,
            "r2_adj": ols_log.rsquared_adj,
            "coef":   float(ols_log.params[-1]),
            "se":     float(ols_log.bse[-1]),
            "p":      float(ols_log.pvalues[-1]),
            "fitted": ols_log.fittedvalues,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "label":  "Logarithmic",
            "x_shifted": x.min() <= 0,
        }
    except Exception as e:
        results["log"] = {"aic": np.nan, "label": "Logarithmic", "error": str(e)}

    # ── 3. Quadratic ──────────────────────────────────────────
    x_sq  = x ** 2
    X_quad = np.column_stack([X_ctrl, x, x_sq])
    try:
        ols_quad = sm.OLS(y, X_quad).fit()
        x_grid   = np.linspace(x.min(), x.max(), 300)
        ctrl_mean = np.full((300, X_ctrl.shape[1]), X_ctrl.mean(axis=0))
        X_pred    = np.column_stack([ctrl_mean, x_grid, x_grid ** 2])
        y_grid    = X_pred @ ols_quad.params
        results["quadratic"] = {
            "aic":      _aic_ols(ols_quad),
            "r2":       ols_quad.rsquared,
            "r2_adj":   ols_quad.rsquared_adj,
            "coef_lin": float(ols_quad.params[-2]),
            "coef_sq":  float(ols_quad.params[-1]),
            "p_sq":     float(ols_quad.pvalues[-1]),  # significance of squared term
            "fitted":   ols_quad.fittedvalues,
            "x_grid":   x_grid,
            "y_grid":   y_grid,
            "label":    "Quadratic",
        }
    except Exception as e:
        results["quadratic"] = {"aic": np.nan, "label": "Quadratic", "error": str(e)}

    # ── 4. GAM ────────────────────────────────────────────────
    if HAS_GAM:
        try:
            # Build feature matrix: controls (linear) + focal predictor (smooth)
            # pyGAM: last column = focal predictor with smooth term
            n_ctrl = X_ctrl.shape[1]
            X_gam  = np.column_stack([X_ctrl, x])
            # Linear terms for controls (indices 0..n_ctrl-1), smooth for last
            terms  = sum([gam_l(i) for i in range(n_ctrl)], gam_l(0))
            # Rebuild: intercept is implicit in pyGAM
            # Simpler: use raw controls + focal predictor
            X_gam2 = np.column_stack([controls, x]) if controls.shape[1] > 0 else x.reshape(-1, 1)
            n_feat = X_gam2.shape[1]
            # Smooth only the last term (focal predictor); linear for controls
            if n_feat == 1:
                term_spec = s(0)
            else:
                ctrl_terms = sum([gam_l(i) for i in range(n_feat - 1)], gam_l(0))
                term_spec  = ctrl_terms + s(n_feat - 1)

            gam = LinearGAM(term_spec).fit(X_gam2, y)

            # Partial dependence of the focal predictor
            x_grid = np.linspace(x.min(), x.max(), 300)
            XX     = gam.generate_X_grid(term=n_feat - 1, n=300)
            pd_mean, pd_ci = gam.partial_dependence(
                term=n_feat - 1, X=XX, width=0.95)

            # Effective degrees of freedom of the smooth (measure of non-linearity)
            edf = float(gam.statistics_["edof_per_coef"][-1]) if hasattr(gam, "statistics_") else np.nan

            # AIC from pyGAM
            gam_aic = float(gam.statistics_["AIC"]) if hasattr(gam, "statistics_") else np.nan

            results["gam"] = {
                "aic":    gam_aic,
                "r2":     float(gam.statistics_["pseudo_r2"]["explained_deviance"])
                          if hasattr(gam, "statistics_") else np.nan,
                "r2_adj": np.nan,
                "edf":    edf,
                "fitted": gam.predict(X_gam2),
                "x_grid": XX[:, -1],        # x values of the partial dependence grid
                "y_grid": pd_mean,           # partial dependence mean
                "y_ci_lo": pd_ci[:, 0],
                "y_ci_hi": pd_ci[:, 1],
                "label":  "GAM smooth",
            }
        except Exception as e:
            results["gam"] = {"aic": np.nan, "label": "GAM smooth", "error": str(e)}
    else:
        results["gam"] = {"aic": np.nan, "label": "GAM smooth (not installed)",
                          "error": "pyGAM not available"}

    return results


# ── Per-predictor plot ────────────────────────────────────────────────────────

def plot_functional_forms(d: pd.DataFrame,
                           health: str,
                           predictors: list,
                           controls: list,
                           sp_pretty: str,
                           outpath: Path) -> list:
    """
    For each predictor: fit all four functional forms and plot them
    overlaid on the raw scatter.  AIC values are annotated.

    Returns a list of result dicts (one per predictor) for the
    combined summary table.
    """
    h_lbl   = HEALTH_LABELS.get(health, health)
    y       = d[health].values.astype(float)
    X_ctrl  = d[controls].values.astype(float) if controls else np.empty((len(d), 0))

    n_pred = len(predictors)
    ncols  = 2
    nrows  = int(np.ceil(n_pred / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(7 * ncols, 5.5 * nrows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    all_rows = []

    for i, pred in enumerate(predictors):
        ax    = axes[i]
        p_lbl = PREDICTOR_LABELS.get(pred, pred)
        x     = d[pred].values.astype(float)

        # Raw scatter
        ax.scatter(x, y, s=10, alpha=0.20, color="grey",
                   linewidths=0, zorder=1, label="_")

        # Fit all forms
        forms = fit_univariate_forms(x, y, X_ctrl)

        # Find best AIC among fitted forms
        aic_vals = {k: v["aic"] for k, v in forms.items() if np.isfinite(v.get("aic", np.nan))}
        best_form = min(aic_vals, key=aic_vals.get) if aic_vals else None
        aic_lin   = forms["linear"].get("aic", np.nan)

        # Plot each form
        for form_key, form_res in forms.items():
            if "x_grid" not in form_res or "y_grid" not in form_res:
                continue
            color   = FORM_COLOURS.get(form_key, "black")
            is_best = (form_key == best_form)
            lw      = 2.2 if is_best else 1.2
            ls      = "-" if is_best else "--"
            daic    = form_res["aic"] - aic_lin if np.isfinite(form_res["aic"]) else np.nan
            lbl     = (f"{form_res['label']}  "
                       f"AIC={form_res['aic']:.0f}  "
                       f"(ΔAIC={daic:+.0f})"
                       + (" ★" if is_best else ""))
            ax.plot(form_res["x_grid"], form_res["y_grid"],
                    color=color, lw=lw, ls=ls, zorder=3, label=lbl)

            # GAM confidence band
            if form_key == "gam" and "y_ci_lo" in form_res:
                ax.fill_between(form_res["x_grid"],
                                form_res["y_ci_lo"],
                                form_res["y_ci_hi"],
                                color=color, alpha=0.12, zorder=2)

        # Annotation box: ΔAIC relative to linear
        ann_lines = [f"ΔAIC vs linear:"]
        for fk in ["log", "quadratic", "gam"]:
            fv = forms.get(fk, {})
            if np.isfinite(fv.get("aic", np.nan)):
                d_aic = fv["aic"] - aic_lin
                ann_lines.append(f"  {fv['label']:12s}: {d_aic:+.0f}")
        if best_form:
            ann_lines.append(f"Best: {forms[best_form]['label']}")

        ax.text(0.03, 0.97, "\n".join(ann_lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=BASE_FONT - 3,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="lightgrey"))

        ax.set_xlabel(p_lbl, fontsize=BASE_FONT)
        ax.set_ylabel(h_lbl, fontsize=BASE_FONT)
        ax.set_title(p_lbl, fontsize=TITLE_FONT, fontweight="bold")
        ax.legend(fontsize=BASE_FONT - 3, loc="lower left", framealpha=0.7)

        # Collect summary row
        row = {
            "predictor":       pred,
            "predictor_label": p_lbl,
            "aic_linear":      forms["linear"].get("aic", np.nan),
            "aic_log":         forms["log"].get("aic", np.nan),
            "aic_quadratic":   forms["quadratic"].get("aic", np.nan),
            "aic_gam":         forms["gam"].get("aic", np.nan),
            "daic_log":        forms["log"].get("aic", np.nan) - forms["linear"].get("aic", np.nan),
            "daic_quadratic":  forms["quadratic"].get("aic", np.nan) - forms["linear"].get("aic", np.nan),
            "daic_gam":        forms["gam"].get("aic", np.nan) - forms["linear"].get("aic", np.nan),
            "best_form":       best_form,
            "r2_linear":       forms["linear"].get("r2", np.nan),
            "r2_log":          forms["log"].get("r2", np.nan),
            "r2_quadratic":    forms["quadratic"].get("r2", np.nan),
            "p_sq_term":       forms["quadratic"].get("p_sq", np.nan),
            "gam_edf":         forms["gam"].get("edf", np.nan),
            "coef_linear":     forms["linear"].get("coef", np.nan),
            "coef_log":        forms["log"].get("coef", np.nan),
            "p_linear":        forms["linear"].get("p", np.nan),
            "p_log":           forms["log"].get("p", np.nan),
        }
        all_rows.append(row)

    for j in range(n_pred, len(axes)):
        axes[j].set_visible(False)

    # Legend for line styles
    legend_patches = [
        mpatches.Patch(color=FORM_COLOURS["linear"],    label="Linear"),
        mpatches.Patch(color=FORM_COLOURS["log"],       label="Logarithmic"),
        mpatches.Patch(color=FORM_COLOURS["quadratic"], label="Quadratic"),
        mpatches.Patch(color=FORM_COLOURS["gam"],       label="GAM smooth"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=BASE_FONT - 1, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"{sp_pretty} – {h_lbl}\n"
        f"Functional form comparison: linear vs log vs quadratic vs GAM\n"
        f"ΔAIC relative to linear model  |  ★ = best-fitting form  |  "
        f"ΔAIC < 0 = improvement over linear",
        fontsize=TITLE_FONT
    )
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    return all_rows


# ── Combined summary plots ────────────────────────────────────────────────────

def plot_winner_heatmap(all_df: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap showing the winning functional form per
    predictor × health metric × species.
    Colour-coded by form.
    """
    FORM_CMAP = {
        "linear":    "#4C72B0",
        "log":       "#DD8452",
        "quadratic": "#55A868",
        "gam":       "#C44E52",
        None:        "#eeeeee",
    }

    species_list = list(all_df["species"].unique())
    health_list  = [HEALTH_LABELS.get(h, h) for h in HEALTH_VARS
                    if HEALTH_LABELS.get(h, h) in all_df["health_metric"].values]
    pred_list    = list(PREDICTOR_LABELS.values())

    n_sp   = len(species_list)
    n_hm   = len(health_list)
    n_pred = len(pred_list)

    sq  = 0.3
    gap = 0.05
    cell_w = n_sp * (sq + gap)

    fig, ax = plt.subplots(
        figsize=(cell_w * n_hm * 0.9 + 2.5, 1.0 + 0.85 * n_pred)
    )

    for pi, pred_lbl in enumerate(pred_list):
        for hi, hm in enumerate(health_list):
            for si, sp in enumerate(species_list):
                sub = all_df[
                    (all_df["health_metric"]    == hm) &
                    (all_df["predictor_label"]  == pred_lbl) &
                    (all_df["species"]          == sp)
                ]
                winner = sub["best_form"].values[0] if len(sub) > 0 else None
                color  = FORM_CMAP.get(winner, "#eeeeee")
                x0 = hi * (cell_w + 0.3) + si * (sq + gap)
                y0 = pi
                ax.add_patch(plt.Rectangle((x0, y0), sq, sq * 0.85,
                                           color=color, zorder=2))

    xtick_pos = [hi * (cell_w + 0.3) + cell_w / 2 for hi in range(n_hm)]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(health_list, rotation=35, ha="right", fontsize=BASE_FONT - 1)
    ax.set_yticks(np.arange(n_pred) + 0.38)
    ax.set_yticklabels(pred_list, fontsize=BASE_FONT - 1)
    ax.set_xlim(-0.3, n_hm * (cell_w + 0.3))
    ax.set_ylim(-0.2, n_pred + 0.3)
    ax.invert_yaxis()
    ax.set_aspect("auto")

    # Species mini-labels
    for si, sp in enumerate(species_list):
        x0 = si * (sq + gap)
        ax.text(x0 + sq / 2, n_pred + 0.08,
                sp.split()[0][0] + "." + sp.split()[-1][:4],
                ha="center", va="top", fontsize=BASE_FONT - 4,
                style="italic", rotation=40)

    legend_patches = [
        mpatches.Patch(color=FORM_CMAP["linear"],    label="Linear"),
        mpatches.Patch(color=FORM_CMAP["log"],       label="Logarithmic"),
        mpatches.Patch(color=FORM_CMAP["quadratic"], label="Quadratic"),
        mpatches.Patch(color=FORM_CMAP["gam"],       label="GAM"),
        mpatches.Patch(color=FORM_CMAP[None],        label="No result"),
    ]
    ax.legend(handles=legend_patches, loc="lower center", ncol=5,
              fontsize=BASE_FONT - 1, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.22))

    ax.set_title(
        "Best-fitting functional form per predictor × health metric × species\n"
        "(selected by lowest AIC)",
        fontsize=TITLE_FONT
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_daic_log_heatmap(all_df: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap of mean ΔAIC(log − linear) per predictor × health metric,
    averaged across species.

    Negative values = log model wins.
    More negative = stronger evidence that the relationship is logarithmic.
    """
    pred_codes  = PREDICTORS
    pred_labels = [PREDICTOR_LABELS.get(p, p) for p in pred_codes]
    health_list = [HEALTH_LABELS.get(h, h) for h in HEALTH_VARS]

    agg = (all_df.groupby(["health_metric", "predictor"])["daic_log"]
           .mean().reset_index())

    mat = np.full((len(pred_codes), len(health_list)), np.nan)
    for _, row in agg.iterrows():
        pi = (pred_codes.index(row["predictor"])
              if row["predictor"] in pred_codes else -1)
        hi = (health_list.index(row["health_metric"])
              if row["health_metric"] in health_list else -1)
        if pi >= 0 and hi >= 0:
            mat[pi, hi] = row["daic_log"]

    # Diverging colormap: blue = log wins (negative), red = log loses (positive)
    vmax = np.nanmax(np.abs(mat)) if np.any(np.isfinite(mat)) else 10

    fig, ax = plt.subplots(figsize=(2.5 * len(health_list),
                                     0.9 + 0.75 * len(pred_codes)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)

    for i in range(len(pred_codes)):
        for j in range(len(health_list)):
            v = mat[i, j]
            if np.isfinite(v):
                flag = " ★" if v < -2 else ""
                dark = abs(v) > vmax * 0.5
                ax.text(j, i, f"{v:+.0f}{flag}",
                        ha="center", va="center", fontsize=BASE_FONT - 2,
                        color="white" if dark else "black")

    ax.set_xticks(range(len(health_list)))
    ax.set_xticklabels(health_list, rotation=35, ha="right", fontsize=BASE_FONT - 1)
    ax.set_yticks(range(len(pred_codes)))
    ax.set_yticklabels(pred_labels, fontsize=BASE_FONT - 1)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("ΔAIC  (log − linear)  |  Blue = log wins  |  Red = linear wins",
                   fontsize=BASE_FONT)

    ax.set_title(
        "ΔAIC: logarithmic vs linear functional form\n"
        "Mean across species  |  ★ = ΔAIC < −2 (meaningful log improvement)  |  "
        "Negative = log model fits better",
        fontsize=TITLE_FONT
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_rows = []

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
            h_out  = sp_out / health
            h_safe = health.replace("/", "_")
            h_lbl  = HEALTH_LABELS.get(health, health)
            ensure_dir(h_out)

            # Filter to complete cases for this health metric
            needed = [health] + PREDICTORS + CONTROL_VARS
            d = df[needed].dropna().copy()
            for c in needed:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna()

            if len(d) < MIN_N:
                print(f"    [skip] n={len(d)} < {MIN_N}")
                continue

            # ── Functional form comparison plot ───────────────
            rows = plot_functional_forms(
                d          = d,
                health     = health,
                predictors = PREDICTORS,
                controls   = CONTROL_VARS,
                sp_pretty  = sp_pretty,
                outpath    = h_out / f"{h_safe}_functional_form_comparison.png",
            )

            # Add metadata and write text summary
            with open(h_out / f"{h_safe}_functional_form_AIC.txt", "w") as f:
                f.write(f"Functional form comparison — {sp_pretty} | {h_lbl}\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"{'Predictor':<35} {'Linear':>8} {'Log':>8} "
                        f"{'Quad':>8} {'GAM':>8} {'Best':>12}\n")
                f.write(f"{'-'*80}\n")
                for row in rows:
                    f.write(
                        f"{row['predictor_label']:<35} "
                        f"{row['aic_linear']:>8.1f} "
                        f"{row['aic_log']:>8.1f} "
                        f"{row['aic_quadratic']:>8.1f} "
                        f"{row['aic_gam']:>8.1f} "
                        f"{str(row['best_form']):>12}\n"
                    )
                f.write(f"\n{'ΔAIC vs linear (negative = improvement)':}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{'Predictor':<35} {'ΔAIC_log':>10} "
                        f"{'ΔAIC_quad':>10} {'ΔAIC_gam':>10}\n")
                for row in rows:
                    f.write(
                        f"{row['predictor_label']:<35} "
                        f"{row['daic_log']:>+10.1f} "
                        f"{row['daic_quadratic']:>+10.1f} "
                        f"{row['daic_gam']:>+10.1f}\n"
                    )
                f.write(
                    "\nInterpretation:\n"
                    "  ΔAIC < -2   : meaningful improvement over linear\n"
                    "  ΔAIC < -10  : strong improvement\n"
                    "  ΔAIC < -100 : overwhelming improvement\n"
                    "  Note: large n inflates AIC differences — "
                    "inspect plots alongside AIC values.\n"
                )

            for row in rows:
                row["species"]       = sp_pretty
                row["health_metric"] = h_lbl
            all_rows.extend(rows)

        print(f"  [ok] → {sp_out}")

    # ── Combined outputs ──────────────────────────────────────
    if all_rows:
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(OUT_ROOT / "ALL_functional_form_AIC.csv", index=False)
        plot_winner_heatmap(all_df,      OUT_ROOT / "ALL_functional_form_winner.png")
        plot_daic_log_heatmap(all_df,    OUT_ROOT / "ALL_log_vs_linear_DAIC.png")

        # Quick console summary for peak NDVI
        peak = all_df[all_df["health_metric"] == "Peak NDVI"]
        if not peak.empty:
            print("\n── Peak NDVI summary ──────────────────────────────────")
            print(peak[["species", "predictor_label",
                         "daic_log", "daic_quadratic", "best_form"]]
                  .sort_values(["species", "predictor_label"])
                  .to_string(index=False))

    print(f"\n[ok] All functional form outputs saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()