#!/usr/bin/env python3
"""
Appendix analysis – two supplementary analyses to accompany the main MLR study.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART A – EXTENDED HEALTH METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs OLS dominance analysis (height-controlled) for health metrics
that were excluded from the main analysis due to high inter-metric
correlation. Results are summarised as:
  - Dominance weight heatmap per predictor across all health metrics
    and species  (AppA_dominance_extended_heatmap.png)
  - Stacked bar chart of average R² partition across all metrics
    (AppA_dominance_extended_stacked.png)
  - Full results table  (AppA_dominance_extended.csv)

PART B – ALTERNATIVE PREDICTORS (EXTRA POLLUTANTS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fits univariate OLS models (health ~ predictor + height) for each
alternative predictor against the five main health metrics. Because
alternative predictors are strongly correlated with the main ones
(same environmental variable, different spatial scale or chemical
proxy), they are NOT added to the existing multivariate models.
Results are summarised as:
  - Coefficient direction × significance heatmap, one panel per species
    (AppB_univariate_coef_heatmap.png)

    Design rationale
    ────────────────
    Colour HUE   : direction of association
                     Red  = positive (β > 0)
                     Blue = negative (β < 0)
                     Grey = zero coefficient / no data

    Colour SHADE : significance of the association (p < 0.05)
                     Full shade  = significant
                     Light shade = not significant

    Cell TEXT    : ✓ where the association is statistically significant
                   (p < 0.05)

    Legend       : discrete patches matching the actual categories shown.

  - R² heatmap showing explained variance per predictor × metric,
    one panel per species, shared colour scale
    (AppB_univariate_r2_heatmap.png)
  - Full results table  (AppB_univariate_results.csv)

Dependencies:
    pip install numpy pandas matplotlib scipy statsmodels geopandas
                scikit-learn shapely pyproj fiona
"""

import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ── Font sizing ───────────────────────────────────────────
# Single knob to scale ALL text in the figures.
#   1.0 = original sizes; increase for larger fonts (1.4 ≈ +40%).
FONT_SCALE = 1.4

def _fs(size):
    """Scale a base font size by the global FONT_SCALE factor."""
    return size * FONT_SCALE

# Scale the matplotlib default too, so elements without an explicit
# fontsize (e.g. colourbar tick labels) grow consistently.
plt.rcParams['font.size'] = 10 * FONT_SCALE
from scipy import stats

import geopandas as gpd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors


# ╔══════════════════════════════════════════════════════════╗
# ║                    USER SETTINGS                        ║
# ╚══════════════════════════════════════════════════════════╝

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

OUT_ROOT = Path(
    "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/"
    "Data analysis/ndvi background investigations/_APPENDIX"
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Shapefile join (same as main script)
SHAPEFILE_PATH             = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/crown_shapes_final_CRS.shp"
CSV_ID_COL                 = "tree_id"
SHP_ID_COL                 = "crown_id"
USE_CENTROID_IF_NOT_POINTS = True
TARGET_CRS                 = None
COORD_COLS                 = ["x", "y"]

# ── Analysis settings ─────────────────────────────────────
CONTROL_VARS = ["height"]
ALPHA        = 0.05
MIN_N        = 30
FIG_DPI      = 200
MAX_PREDICTORS_DA = 10

# Number of species in the study – used to scale directional-agreement
# colour shading. Update this if species are added or removed.
N_SPECIES = 5

# ── MAIN health metrics (used in main analysis) ───────────
MAIN_HEALTH_VARS = [
    "ndvi_peak",
    "sos_doy",
    "los_days",
    "amplitude",
    "auc_above_base_full",
]

# ── MAIN predictors (used in main analysis) ───────────────
MAIN_PREDICTORS = [
    "imperv_100m",
    "poll_bc_anmean",
    "lst_temp_r100_y",
    "insolation9",
]

# ── PART A: additional health metrics for appendix ────────
EXTRA_HEALTH_VARS = [
    "greenup_doy",
    "peak_doy",
    "sen_onset_doy",
    "eos_doy",
    "ndvi_base",
    "ndvi_eos",
    "plateau_days",
    "decline_days",
    "slope_sos_peak",
    "senescence_rate",
    "auc_above_base_sos_eos",
]

# ── PART B: alternative predictors for appendix ──────────
EXTRA_PREDICTORS = {
    "poll_no2_anmean":    "NO₂ (annual mean)",
    "poll_pm10_anmean":   "PM10 (annual mean)",
    "poll_pm25_anmean":   "PM2.5 (annual mean)",
    "poll_belaqi_anmean": "BelAQI index (annual mean)",
}

# ── Human-readable labels ─────────────────────────────────
HEALTH_LABELS = {
    # Main
    "ndvi_peak":              "Peak NDVI",
    "sos_doy":                "Start of season (DOY)",
    "los_days":               "Length of season (days)",
    "amplitude":              "NDVI amplitude",
    "auc_above_base_full":    "Seasonal NDVI integral",
    # Extra
    "greenup_doy":            "Green-up (DOY)",
    "peak_doy":               "Peak (DOY)",
    "sen_onset_doy":          "Senescence onset (DOY)",
    "eos_doy":                "End of season (DOY)",
    "ndvi_base":              "Base NDVI",
    "ndvi_eos":               "EOS NDVI",
    "plateau_days":           "Plateau duration (days)",
    "decline_days":           "Decline duration (days)",
    "slope_sos_peak":         "Green-up slope",
    "senescence_rate":        "Senescence rate",
    "auc_above_base_sos_eos": "Seasonal integral (SOS–EOS)",
}

PREDICTOR_LABELS = {
    "imperv_100m":        "Impervious surface",
    "poll_bc_anmean":     "Black carbon",
    "lst_temp_r100_y":    "LST",
    "insolation9":        "Solar radiation",
}
PREDICTOR_LABELS.update(EXTRA_PREDICTORS)

# ── Biological filters ────────────────────────────────────
BIOLOGICAL_FILTERS = {
    "los_days":               lambda x: (x >= 60)   & (x <= 365),
    "sos_doy":                lambda x: (x >= 1)    & (x <= 250),
    "greenup_doy":            lambda x: (x >= 1)    & (x <= 200),
    "peak_doy":               lambda x: (x >= 1)    & (x <= 300),
    "sen_onset_doy":          lambda x: (x >= 150)  & (x <= 365),
    "eos_doy":                lambda x: (x >= 200)  & (x <= 365),
    "ndvi_peak":              lambda x: (x >= -0.1) & (x <= 1.0),
    "amplitude":              lambda x: (x >= -0.1) & (x <= 1.0),
    "ndvi_base":              lambda x: (x >= -0.1) & (x <= 1.0),
    "ndvi_eos":               lambda x: (x >= -0.1) & (x <= 1.0),
    "auc_above_base_full":    lambda x: x > -1e9,
    "auc_above_base_sos_eos": lambda x: x > -1e9,
    "plateau_days":           lambda x: (x >= 0)    & (x <= 365),
    "decline_days":           lambda x: (x >= 0)    & (x <= 365),
    "height":                 lambda x: x > 1,
    "poll_bc_anmean":         lambda x: x > 0,
    "poll_no2_anmean":        lambda x: x > 0,
    "poll_pm10_anmean":       lambda x: x > 0,
    "poll_pm25_anmean":       lambda x: x > 0,
}

# Colour palette for dominance stacked bars
DA_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
              "#CCB974", "#64B5CD", "#B07AA1", "#FF9DA7"]

# Consistent health metric order across all figures
HEALTH_ORDER_MAIN  = [HEALTH_LABELS[h] for h in MAIN_HEALTH_VARS]
HEALTH_ORDER_EXTRA = [HEALTH_LABELS[h] for h in EXTRA_HEALTH_VARS]
HEALTH_ORDER_ALL   = HEALTH_ORDER_MAIN + HEALTH_ORDER_EXTRA

# ── Colours used in Part B coefficient heatmap ────────────
# Base hues for positive (red) and negative (blue) directions.
# Two shades per hue correspond to significant / not significant.
_RED_BASE  = np.array([0.769, 0.306, 0.322])   # #C44E52
_BLUE_BASE = np.array([0.298, 0.447, 0.690])   # #4C72B0
_GREY_CELL = np.array([0.88,  0.88,  0.88])    # no data / zero coef

def _shade(base_rgb, significant):
    """
    Return an RGBA tuple for a single-species cell.
      significant=True  → full colour  (alpha 1.0)
      significant=False → light shade  (alpha 0.35)
    """
    alpha   = 1.0 if significant else 0.35
    white   = np.ones(3)
    blended = alpha * base_rgb + (1 - alpha) * white
    return tuple(blended) + (1.0,)


# ╔══════════════════════════════════════════════════════════╗
# ║                      UTILITIES                          ║
# ╚══════════════════════════════════════════════════════════╝

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


def load_species(species, csv_path, required_cols):
    """Load, join coordinates, clean and filter one species CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [skip] Cannot read CSV: {e}")
        return None

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        hard_missing = [c for c in missing if c in MAIN_PREDICTORS + CONTROL_VARS]
        if hard_missing:
            print(f"  [skip] Missing required columns: {hard_missing}")
            return None

    try:
        df = attach_xy_from_shapefile(df)
    except Exception as e:
        print(f"  [skip] Coordinate join failed: {e}")
        return None

    if not all(c in df.columns for c in COORD_COLS):
        print("  [skip] Coordinate columns missing.")
        return None
    if df[COORD_COLS].isna().all().all():
        print("  [skip] All-NaN coordinates (ID mismatch?).")
        return None

    numeric_cols = [c for c in required_cols if c in df.columns] + COORD_COLS
    df = make_numeric(df, numeric_cols)
    df = apply_filters(df, numeric_cols)
    print(f"  n = {len(df)} trees after filtering")
    return df


# ╔══════════════════════════════════════════════════════════╗
# ║              DOMINANCE ANALYSIS (OLS)                   ║
# ╚══════════════════════════════════════════════════════════╝

def _r2_ols(y, X):
    if X.shape[1] == 0:
        return 0.0
    try:
        Xc  = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()
        return max(res.rsquared, 0.0)
    except Exception:
        return 0.0


def dominance_analysis(y, X_predictors, X_controls, predictor_names):
    """
    General dominance weights (Budescu, 1993) after partialling out controls.
    Returns DataFrame with columns: predictor, dominance_weight, pct_of_r2.
    """
    k = len(predictor_names)
    if k > MAX_PREDICTORS_DA:
        raise ValueError(f"Too many predictors ({k}).")

    def _resid_ctrl(v):
        if X_controls.shape[1] == 0:
            return v - v.mean()
        Xc = sm.add_constant(X_controls, has_constant="add")
        return sm.OLS(v, Xc).fit().resid

    y_r  = _resid_ctrl(y)
    Xp_r = np.column_stack([_resid_ctrl(X_predictors[:, j]) for j in range(k)])

    r2_cache = {frozenset(): 0.0}
    for size in range(1, k + 1):
        for subset in combinations(range(k), size):
            fs = frozenset(subset)
            r2_cache[fs] = _r2_ols(y_r, Xp_r[:, list(subset)])

    weights = np.zeros(k)
    for j in range(k):
        avgs = []
        for s in range(k):
            others = [i for i in range(k) if i != j]
            incs = [r2_cache[frozenset(sub) | {j}] - r2_cache[frozenset(sub)]
                    for sub in combinations(others, s)]
            if incs:
                avgs.append(np.mean(incs))
        weights[j] = np.mean(avgs)

    total = weights.sum()
    pct   = (weights / total * 100) if total > 1e-10 else np.zeros(k)

    return pd.DataFrame({
        "predictor":        [PREDICTOR_LABELS.get(p, p) for p in predictor_names],
        "predictor_code":   predictor_names,
        "dominance_weight": weights,
        "pct_of_r2":        pct,
    }).sort_values("dominance_weight", ascending=False).reset_index(drop=True)


# ╔══════════════════════════════════════════════════════════╗
# ║        PART A – EXTENDED HEALTH METRICS                 ║
# ╚══════════════════════════════════════════════════════════╝

def run_part_A():
    """
    Dominance analysis on additional health metrics using the same
    four main predictors. One row per species × health metric × predictor.
    """
    print("\n" + "="*60)
    print("  PART A – Extended health metrics (dominance analysis)")
    print("="*60)

    all_da_rows = []
    required = list(set(EXTRA_HEALTH_VARS + MAIN_PREDICTORS + CONTROL_VARS))

    for species, csv_path in SPECIES_CSVS.items():
        print(f"\n  {SPECIES_LABELS[species]}")
        sp_pretty = SPECIES_LABELS[species]

        df = load_species(species, csv_path, required)
        if df is None:
            continue

        for health in EXTRA_HEALTH_VARS:
            if health not in df.columns:
                print(f"    [skip] {health}: column not found")
                continue

            h_lbl  = HEALTH_LABELS.get(health, health)
            needed = [health] + MAIN_PREDICTORS + CONTROL_VARS
            d = df[[c for c in needed if c in df.columns]].dropna().copy()
            for c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna()

            if len(d) < MIN_N:
                print(f"    [skip] {health}: n={len(d)} < {MIN_N}")
                continue

            y          = d[health].values.astype(float)
            X_pred_arr = d[MAIN_PREDICTORS].values.astype(float)
            X_ctrl_arr = d[CONTROL_VARS].values.astype(float)

            try:
                da_df = dominance_analysis(y, X_pred_arr, X_ctrl_arr, MAIN_PREDICTORS)
            except Exception as e:
                print(f"    [error] {health}: {e}")
                continue

            da_df["species"]       = sp_pretty
            da_df["health_metric"] = h_lbl
            da_df["n"]             = len(d)
            all_da_rows.extend(da_df.to_dict("records"))
            print(f"    ✓ {h_lbl}  (n={len(d)})")

    if not all_da_rows:
        print("  [warn] No results for Part A.")
        return

    da_all = pd.DataFrame(all_da_rows)
    da_all.to_csv(OUT_ROOT / "AppA_dominance_extended.csv", index=False)
    print(f"\n  Saved: AppA_dominance_extended.csv")

    _plot_A_heatmap(da_all)
    _plot_A_stacked(da_all)


def _plot_A_heatmap(da_all):
    """
    Heatmap of dominance weights (% of R²) per predictor × phenological metric,
    averaged across species. One panel per predictor.
    """
    predictors_pretty = [PREDICTOR_LABELS.get(p, p) for p in MAIN_PREDICTORS]
    health_order      = [h for h in HEALTH_ORDER_EXTRA
                         if h in da_all["health_metric"].unique()]
    species_list      = list(da_all["species"].unique())

    n_pred = len(predictors_pretty)
    fig, axes = plt.subplots(
        1, n_pred,
        figsize=(4.5 * n_pred, 1.2 + 0.58 * len(species_list)),
        sharey=True
    )
    if n_pred == 1:
        axes = [axes]

    for ax, pred_code, pred_pretty in zip(axes, MAIN_PREDICTORS, predictors_pretty):
        sub = da_all[da_all["predictor"] == pred_pretty].copy()
        piv = (sub.pivot_table(index="species", columns="health_metric",
                               values="pct_of_r2", aggfunc="mean")
               .reindex(columns=health_order))

        im = ax.imshow(piv.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(health_order)))
        ax.set_xticklabels(health_order, rotation=40, ha="right", fontsize=_fs(7))
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=_fs(8))
        ax.set_title(pred_pretty, fontsize=_fs(9))

        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.0f}%",
                            ha="center", va="center", fontsize=_fs(7),
                            color="white" if val > 55 else "black")
        plt.colorbar(im, ax=ax, label="% of R²", shrink=0.7)

    fig.suptitle(
        "Appendix A – Dominance analysis: extended phenological metrics\n"
        "% of R² per predictor × phenological metric (averaged across species)",
        fontsize=_fs(11)
    )
    plt.tight_layout()
    fig.savefig(OUT_ROOT / "AppA_dominance_extended_heatmap.png", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved: AppA_dominance_extended_heatmap.png")


def _plot_A_stacked(da_all):
    """
    Stacked bar chart: average dominance weight per health metric,
    bars coloured by predictor, one bar per health metric.
    """
    pred_list    = [PREDICTOR_LABELS.get(p, p) for p in MAIN_PREDICTORS]
    health_order = [h for h in HEALTH_ORDER_ALL
                    if h in da_all["health_metric"].unique()]

    agg = (da_all.groupby(["health_metric", "predictor"])["dominance_weight"]
           .mean().reset_index())

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(health_order)), 5))
    x       = np.arange(len(health_order))
    bottoms = np.zeros(len(health_order))

    for pi, pred in enumerate(pred_list):
        vals = np.array([
            agg.loc[(agg.health_metric == hm) & (agg.predictor == pred),
                    "dominance_weight"].values[0]
            if len(agg.loc[(agg.health_metric == hm) & (agg.predictor == pred)]) else 0.0
            for hm in health_order
        ])
        ax.bar(x, vals, bottom=bottoms, width=0.6,
               color=DA_COLOURS[pi % len(DA_COLOURS)], label=pred)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(health_order, rotation=35, ha="right", fontsize=_fs(8))
    ax.set_ylabel("Average dominance weight (R²)  [height-controlled]")
    ax.set_title(
        "Appendix A – Dominance analysis: extended phenological metrics\n"
        "Average R² partition across species (main predictors)"
    )
    ax.legend(loc="upper right", fontsize=_fs(8), framealpha=0.85)

    # Vertical separator between main and extra metrics
    if health_order:
        n_extra = len([h for h in health_order if h in HEALTH_ORDER_EXTRA])
        n_main  = len(health_order) - n_extra
        if n_main > 0 and n_extra > 0:
            ax.axvline(n_main - 0.5, color="grey", lw=1.2, ls="--", alpha=0.7)
            ax.text(n_main - 0.5, ax.get_ylim()[1] * 0.97,
                    "← main analysis  |  appendix →",
                    ha="center", va="top", fontsize=_fs(7), color="grey")

    plt.tight_layout()
    fig.savefig(OUT_ROOT / "AppA_dominance_extended_stacked.png", dpi=FIG_DPI)
    plt.close(fig)
    print("  Saved: AppA_dominance_extended_stacked.png")


# ╔══════════════════════════════════════════════════════════╗
# ║        PART B – ALTERNATIVE PREDICTORS                  ║
# ╚══════════════════════════════════════════════════════════╝

def run_part_B():
    """
    Univariate OLS (health ~ predictor + height) for each alternative
    predictor against the five main health metrics.
    Records coefficient, p-value, partial R² and full model R².
    """
    print("\n" + "="*60)
    print("  PART B – Alternative predictors (univariate OLS)")
    print("="*60)

    all_rows = []
    extra_pred_codes = list(EXTRA_PREDICTORS.keys())
    required = list(set(MAIN_HEALTH_VARS + extra_pred_codes + CONTROL_VARS))

    for species, csv_path in SPECIES_CSVS.items():
        print(f"\n  {SPECIES_LABELS[species]}")
        sp_pretty = SPECIES_LABELS[species]

        df = load_species(species, csv_path, required)
        if df is None:
            continue

        for health in MAIN_HEALTH_VARS:
            h_lbl = HEALTH_LABELS.get(health, health)

            for pred_code, pred_label in EXTRA_PREDICTORS.items():
                if pred_code not in df.columns:
                    continue
                if health not in df.columns:
                    continue

                needed = [health, pred_code] + CONTROL_VARS
                d = df[[c for c in needed if c in df.columns]].dropna().copy()
                for c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
                d = d.dropna()

                if len(d) < MIN_N:
                    continue

                y    = d[health].values.astype(float)
                X_df = d[[pred_code] + CONTROL_VARS].copy()
                X_sm = sm.add_constant(X_df, has_constant="add")

                try:
                    res = sm.OLS(y, X_sm).fit()
                except Exception:
                    continue

                # Partial R²: R² drop when predictor is removed
                X_ctrl_only = sm.add_constant(d[CONTROL_VARS], has_constant="add")
                res_ctrl    = sm.OLS(y, X_ctrl_only).fit()
                partial_r2  = res.rsquared - res_ctrl.rsquared

                coef = res.params.get(pred_code, np.nan)
                pval = res.pvalues.get(pred_code, np.nan)
                se   = res.bse.get(pred_code, np.nan)

                all_rows.append({
                    "species":        sp_pretty,
                    "health_metric":  h_lbl,
                    "predictor":      pred_label,
                    "predictor_code": pred_code,
                    "coefficient":    coef,
                    "se":             se,
                    "p_value":        pval,
                    "significant":    pval < ALPHA if np.isfinite(pval) else False,
                    "coef_sign":      int(np.sign(coef)) if np.isfinite(coef) else 0,
                    "R2_model":       res.rsquared,
                    "R2_adj_model":   res.rsquared_adj,
                    "partial_R2":     partial_r2,
                    "n":              len(d),
                })

            print(f"    ✓ {h_lbl}")

    if not all_rows:
        print("  [warn] No results for Part B.")
        return

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT_ROOT / "AppB_univariate_results.csv", index=False)
    print(f"\n  Saved: AppB_univariate_results.csv")

    _plot_B_coef_heatmap(results)
    _plot_B_r2_heatmap(results)


# ──────────────────────────────────────────────────────────
# Part B coefficient heatmap – per species, small multiples
# ──────────────────────────────────────────────────────────

def _plot_B_coef_heatmap(results):
    """
    Coefficient direction heatmap – one panel per species (small multiples).

    Colour hue    : direction of association
                      Red  = positive (β > 0)
                      Blue = negative (β < 0)
                      Grey = zero coefficient / no data
    Colour shade  : significance of the association (p < 0.05)
                      Full shade  = significant
                      Light shade = not significant
    Cell text     : ✓ where the association is statistically significant
    Legend        : discrete patches matching the actual categories shown
    """
    health_order = [h for h in HEALTH_ORDER_MAIN
                    if h in results["health_metric"].unique()]
    pred_order   = [EXTRA_PREDICTORS[p] for p in EXTRA_PREDICTORS
                    if EXTRA_PREDICTORS[p] in results["predictor"].unique()]
    species_list = [SPECIES_LABELS[s] for s in SPECIES_LABELS
                    if SPECIES_LABELS[s] in results["species"].unique()]

    if not health_order or not pred_order or not species_list:
        print("  [warn] No data to plot for Part B coefficient heatmap.")
        return

    n_h = len(health_order)
    n_p = len(pred_order)
    n_s = len(species_list)

    fig, axes = plt.subplots(
        1, n_s,
        figsize=(max(4, 1.4 * n_h) * n_s, 1.8 + 0.65 * n_p),
        sharey=True
    )
    if n_s == 1:
        axes = [axes]

    for ax, species in zip(axes, species_list):
        sp_results = results[results["species"] == species]

        for pi, pred in enumerate(pred_order):
            for hi, hm in enumerate(health_order):
                row = sp_results[(sp_results["predictor"] == pred) &
                                 (sp_results["health_metric"] == hm)]

                if row.empty:
                    facecolor = tuple(_GREY_CELL) + (1.0,)
                    txt       = ""
                else:
                    row       = row.iloc[0]
                    coef_sign = row["coef_sign"]
                    is_sig    = bool(row["significant"])

                    if coef_sign > 0:
                        facecolor = _shade(_RED_BASE,  is_sig)
                    elif coef_sign < 0:
                        facecolor = _shade(_BLUE_BASE, is_sig)
                    else:
                        facecolor = tuple(_GREY_CELL) + (1.0,)

                    txt = "✓" if is_sig else ""

                rect = plt.Rectangle(
                    (hi - 0.5, pi - 0.5), 1, 1,
                    facecolor=facecolor, edgecolor="white", linewidth=1.2
                )
                ax.add_patch(rect)

                if txt:
                    brightness = (0.299 * facecolor[0] +
                                  0.587 * facecolor[1] +
                                  0.114 * facecolor[2])
                    txt_color  = "white" if brightness < 0.55 else "black"
                    ax.text(hi, pi, txt,
                            ha="center", va="center",
                            fontsize=_fs(10), color=txt_color, fontweight="bold")

        ax.set_xlim(-0.5, n_h - 0.5)
        ax.set_ylim(-0.5, n_p - 0.5)
        ax.set_xticks(range(n_h))
        ax.set_xticklabels(health_order, rotation=35, ha="right", fontsize=_fs(8))
        ax.set_aspect("equal")
        ax.set_title(species, fontsize=_fs(9), style="italic")

    axes[0].set_yticks(range(n_p))
    axes[0].set_yticklabels(pred_order, fontsize=_fs(9))

    # ── Discrete legend ───────────────────────────────────
    legend_items = [
        mpatches.Patch(
            facecolor=_shade(_RED_BASE,  True),
            edgecolor="grey", linewidth=0.5, label="Positive, significant (p < 0.05)"
        ),
        mpatches.Patch(
            facecolor=_shade(_RED_BASE,  False),
            edgecolor="grey", linewidth=0.5, label="Positive, not significant"
        ),
        mpatches.Patch(
            facecolor=_shade(_BLUE_BASE, True),
            edgecolor="grey", linewidth=0.5, label="Negative, significant (p < 0.05)"
        ),
        mpatches.Patch(
            facecolor=_shade(_BLUE_BASE, False),
            edgecolor="grey", linewidth=0.5, label="Negative, not significant"
        ),
        mpatches.Patch(
            facecolor=tuple(_GREY_CELL) + (1.0,),
            edgecolor="grey", linewidth=0.5, label="Zero coefficient / no data"
        ),
    ]
    axes[-1].legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=_fs(8),
        title="Direction & significance\n(✓ = p < 0.05)",
        title_fontsize=_fs(8),
        framealpha=0.9,
    )

    fig.suptitle(
        "Appendix B – Alternative predictors: direction of association per species\n"
        "Colour = direction  |  shade = significance  |  ✓ = p < 0.05",
        fontsize=_fs(10), y=1.01
    )

    plt.tight_layout()
    fig.savefig(OUT_ROOT / "AppB_univariate_coef_heatmap.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: AppB_univariate_coef_heatmap.png")


def _plot_B_r2_heatmap(results):
    """
    Partial R² heatmap – one panel per species, shared colour scale
    (vmax = 95th percentile across all species).
    """
    health_order = [h for h in HEALTH_ORDER_MAIN
                    if h in results["health_metric"].unique()]
    pred_order   = [EXTRA_PREDICTORS[p] for p in EXTRA_PREDICTORS
                    if EXTRA_PREDICTORS[p] in results["predictor"].unique()]
    species_list = [SPECIES_LABELS[s] for s in SPECIES_LABELS
                    if SPECIES_LABELS[s] in results["species"].unique()]

    if not health_order or not pred_order or not species_list:
        print("  [warn] No data to plot for Part B R² heatmap.")
        return

    n_h = len(health_order)
    n_p = len(pred_order)
    n_s = len(species_list)

    # Build all matrices first so we can compute a shared vmax
    mats = {}
    for species in species_list:
        sp_results = results[results["species"] == species]
        mat = np.full((n_p, n_h), np.nan)
        for pi, pred in enumerate(pred_order):
            for hi, hm in enumerate(health_order):
                sub = sp_results[(sp_results["predictor"] == pred) &
                                 (sp_results["health_metric"] == hm)]
                if not sub.empty:
                    mat[pi, hi] = sub["partial_R2"].values[0]
        mats[species] = mat

    all_vals = np.concatenate([m.ravel() for m in mats.values()])
    finite_vals = all_vals[np.isfinite(all_vals)]
    vmax = np.percentile(finite_vals, 95) if len(finite_vals) > 0 else 0.1

    fig, axes = plt.subplots(
        1, n_s,
        figsize=(max(3.5, 1.3 * n_h) * n_s, 1.2 + 0.55 * n_p),
        sharey=True
    )
    if n_s == 1:
        axes = [axes]

    ims = []
    for ax, species in zip(axes, species_list):
        mat = mats[species]
        im  = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ims.append(im)

        for pi in range(n_p):
            for hi in range(n_h):
                val = mat[pi, hi]
                if np.isfinite(val):
                    txt_c = "white" if val > vmax * 0.65 else "black"
                    ax.text(hi, pi, f"{val:.3f}",
                            ha="center", va="center", fontsize=_fs(7.5), color=txt_c)

        ax.set_xticks(range(n_h))
        ax.set_xticklabels(health_order, rotation=35, ha="right", fontsize=_fs(8))
        ax.set_title(species, fontsize=_fs(9), style="italic")

    axes[0].set_yticks(range(n_p))
    axes[0].set_yticklabels(pred_order, fontsize=_fs(8))

    # Single shared colourbar attached to the rightmost panel
    cbar = fig.colorbar(ims[-1], ax=axes[-1], shrink=0.7, pad=0.02)
    cbar.set_label("Partial R²  (above height alone)", fontsize=_fs(8))

    fig.suptitle(
        "Appendix B – Alternative predictors: partial R² per species\n"
        "(univariate OLS, height-controlled; shared colour scale)",
        fontsize=_fs(10)
    )
    plt.tight_layout()
    fig.savefig(OUT_ROOT / "AppB_univariate_r2_heatmap.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: AppB_univariate_r2_heatmap.png")


# ╔══════════════════════════════════════════════════════════╗
# ║                        MAIN                             ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    print("\nAppendix analysis")
    print("─" * 60)
    print(f"Output directory: {OUT_ROOT}\n")

    run_part_A()
    run_part_B()

    print("\n" + "="*60)
    print(f"  Done. All outputs saved to:\n  {OUT_ROOT}")
    print("="*60)


if __name__ == "__main__":
    main()