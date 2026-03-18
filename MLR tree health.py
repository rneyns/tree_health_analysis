#!/usr/bin/env python3
"""
Multi-species Multivariate Linear Regression (MLR) workflow
with Dominance Analysis

For each species CSV and each health metric:
1)  Attach coordinates from shapefile (if not already in CSV)
2)  Apply biological filters and clean data
3)  Fit MLR:  health_metric ~ predictors + height (control)
4)  Diagnostics:
        a) Scatterplots of each predictor vs health metric (with regression line)
        b) Residual normality  (histogram + Q-Q plot + Shapiro-Wilk test)
        c) VIF (variance inflation factor) for multicollinearity
        d) Moran's I on residuals (KNN weights + permutation p-value)
5)  Dominance analysis:
        - Fits all 2^k submodels of the predictors (excluding the control)
        - Computes additional R² contributed by each predictor averaged
          across all subset sizes and orderings (Budescu, 1993; Azen & Budescu, 2003)
        - Reports total, unique, and average additional dominance per predictor
        - Saves dominance table and stacked bar chart per species x health metric
        - Combined heatmap of dominance weights across all species
6)  Save results tables + figures per species
7)  Combined summary tables and figures across all species

Dominance analysis key concepts:
    - General dominance weight  = predictor's average additional R² across all subset models
    - Weights sum to model R² (the part explained by predictors, excluding control)
    - Predictor A generally dominates predictor B if its weight > B's weight in ALL subset sizes
    - More interpretable than β coefficients under multicollinearity (Grömping, 2006)

References:
    Budescu, D. V. (1993). Dominance analysis: A new approach to the problem of
        relative importance of predictors in multiple regression. Psychological Bulletin, 114(3), 542-551.
    Azen, R., & Budescu, D. V. (2003). The dominance analysis approach for comparing
        predictors in multiple regression. Psychological Methods, 8(2), 129-148.
    Grömping, U. (2006). Relative importance for linear regression in R: The package relaimpo.
        Journal of Statistical Software, 17(1), 1-27.
    O'Brien, R. M. (2007). A caution regarding rules of thumb for variance inflation factors.
        Quality & Quantity, 41(5), 673-690.
    Dormann, C. F., et al. (2013). Collinearity: a review of methods to deal with it
        and a simulation study evaluating their performance. Ecography, 36(1), 27-46.

Dependencies:
    pip install numpy pandas matplotlib scipy scikit-learn statsmodels geopandas shapely pyproj fiona
"""

import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

import geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.neighbors import NearestNeighbors


# ============================================================
# USER SETTINGS  –  edit these paths and options
# ============================================================

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

OUT_ROOT = Path("/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/_MLR_MULTI_SPECIES_v2")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Model variables
CONTROL_VARS = ["height"]           # controlled for but excluded from dominance analysis
HEALTH_VARS  = ["ndvi_peak", "sos_doy", "los_days", "amplitude", "auc_above_base_full"]
PREDICTORS   = [                    # 100 m buffers only; one variable per environmental category
    "imperv_100m",
    "poll_bc_anmean",
    "lst_temp_r100_y",
    "insolation9",
]

# Coordinate columns (must exist in CSV or be joined from shapefile)
COORD_COLS = ["x", "y"]

# Shapefile join settings
SHAPEFILE_PATH             = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/crown_shapes_final_CRS.shp"
CSV_ID_COL                 = "tree_id"
SHP_ID_COL                 = "crown_id"
USE_CENTROID_IF_NOT_POINTS = True
TARGET_CRS                 = None   # e.g. "EPSG:31370" or "EPSG:32631"

# Data quality
MIN_N  = 30     # minimum observations to fit a model
ALPHA  = 0.05   # significance threshold

# Moran's I settings
K_NEIGHBORS    = 8
N_PERMUTATIONS = 999
SEED_MORAN     = 42

# Dominance analysis: cap submodel fits for large predictor sets (2^k models total)
# With k=4 predictors this is 15 submodels — fast. Increase MAX_PREDICTORS_DA
# only if you reduce PREDICTORS to avoid combinatorial explosion.
MAX_PREDICTORS_DA = 10

# Plot settings
FIG_DPI = 200

BIOLOGICAL_FILTERS = {
    "los_days":            lambda x: (x >= 60)   & (x <= 365),
    "sos_doy":             lambda x: (x >= 1)    & (x <= 250),
    "ndvi_peak":           lambda x: (x >= -0.1) & (x <= 1.0),
    "amplitude":           lambda x: (x >= -0.1) & (x <= 1.0),
    "auc_above_base_full": lambda x: x > -1e9,
    "height":              lambda x: x > 1,
    "poll_bc_anmean":      lambda x: x > 0,
}

# Colour palette for dominance stacked bars (one per predictor)
DA_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
              "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_filters(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in BIOLOGICAL_FILTERS and c in out.columns:
            out = out[BIOLOGICAL_FILTERS[c](out[c])]
    return out


def _normalize_id_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    frac = np.abs(x - np.round(x))
    x = x.where((~x.isna()) & (frac < 1e-9), x)
    return np.round(x).astype("Int64")


def _normalize_id_str(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"\s+", "", regex=True)
    out = out.str.lower()
    return out.replace("nan", pd.NA)


def attach_xy_from_shapefile(df: pd.DataFrame) -> pd.DataFrame:
    """Attach x/y coordinates from shapefile via ID join (robust to type mismatches)."""
    if all(c in df.columns for c in COORD_COLS):
        return df

    if SHAPEFILE_PATH is None or str(SHAPEFILE_PATH).strip() == "":
        raise ValueError(f"CSV is missing coords {COORD_COLS} and SHAPEFILE_PATH is not set.")

    if CSV_ID_COL not in df.columns:
        raise ValueError(f"CSV is missing join key '{CSV_ID_COL}'.")

    gdf = gpd.read_file(SHAPEFILE_PATH)
    if SHP_ID_COL not in gdf.columns:
        raise ValueError(f"Shapefile is missing join key '{SHP_ID_COL}'.")

    if TARGET_CRS is not None:
        gdf = gdf.to_crs(TARGET_CRS)

    geom = gdf.geometry.centroid if USE_CENTROID_IF_NOT_POINTS else gdf.geometry
    xname, yname = COORD_COLS

    gdf_xy = gdf[[SHP_ID_COL]].copy()
    gdf_xy[xname] = geom.x
    gdf_xy[yname] = geom.y

    df2 = df.copy()
    df2["_jn"]    = _normalize_id_numeric(df2[CSV_ID_COL])
    gdf_xy["_jn"] = _normalize_id_numeric(gdf_xy[SHP_ID_COL])

    merged = df2.merge(gdf_xy[["_jn", xname, yname]], how="left", on="_jn").drop(columns=["_jn"])

    if merged[[xname, yname]].notna().all(axis=1).sum() == 0:
        df2["_js"]    = _normalize_id_str(df2[CSV_ID_COL])
        gdf_xy["_js"] = _normalize_id_str(gdf_xy[SHP_ID_COL])
        merged = df2.merge(
            gdf_xy[["_js", xname, yname]], how="left", on="_js"
        ).drop(columns=["_jn", "_js"], errors="ignore")

    return merged


# ============================================================
# SPATIAL AUTOCORRELATION  (Moran's I)
# ============================================================

def knn_weights(xy: np.ndarray, k: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nbrs.kneighbors(xy)
    idx = idx[:, 1:]
    n = xy.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, idx[i]] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    return W


def morans_I(resid: np.ndarray, W: np.ndarray) -> float:
    z = resid - np.nanmean(resid)
    n = z.size
    den = z @ z
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return (n / W.sum()) * (z @ (W @ z) / den)


def morans_I_perm(resid: np.ndarray, W: np.ndarray,
                  n_perm: int = 999, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    I_obs = morans_I(resid, W)
    if not np.isfinite(I_obs):
        return I_obs, np.nan
    sims = np.array([morans_I(rng.permutation(resid), W) for _ in range(n_perm)])
    p = (np.sum(np.abs(sims) >= np.abs(I_obs)) + 1) / (n_perm + 1)
    return I_obs, p


# ============================================================
# DOMINANCE ANALYSIS
# ============================================================

def _r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    """Fit OLS and return R². Returns 0 for degenerate cases."""
    if X.shape[1] == 0:
        return 0.0
    try:
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()
        return max(res.rsquared, 0.0)
    except Exception:
        return 0.0


def dominance_analysis(y: np.ndarray,
                        X_predictors: np.ndarray,
                        X_controls: np.ndarray,
                        predictor_names: list) -> pd.DataFrame:
    """
    Compute general dominance weights (Budescu, 1993) for each predictor
    after partialling out the control variables.

    Strategy:
        1. Residualise y and each predictor column on the controls.
           This removes the control variance before dominance analysis,
           so weights reflect contributions among trees of similar height.
        2. For every non-empty subset S of predictors, fit OLS on the
           residualised predictors in S and record R²(S).
        3. For each predictor j and each subset S not containing j,
           compute the additional R² from adding j: R²(S∪{j}) − R²(S).
        4. Average the additional R² over all subsets of the same size,
           then average across all sizes → general dominance weight.

    Returns a DataFrame with columns:
        predictor, dominance_weight, pct_of_r2
    sorted descending by dominance_weight.
    """
    k = len(predictor_names)
    if k > MAX_PREDICTORS_DA:
        raise ValueError(
            f"Too many predictors for dominance analysis ({k}). "
            f"Reduce PREDICTORS or increase MAX_PREDICTORS_DA."
        )

    # ---- Residualise on controls ----
    def _resid_on_controls(v: np.ndarray) -> np.ndarray:
        if X_controls.shape[1] == 0:
            return v - v.mean()
        Xc = sm.add_constant(X_controls, has_constant="add")
        return sm.OLS(v, Xc).fit().resid

    y_resid = _resid_on_controls(y)
    Xp_resid = np.column_stack([_resid_on_controls(X_predictors[:, j]) for j in range(k)])

    # ---- Cache R² for all subsets ----
    # key: frozenset of predictor indices → R²
    r2_cache: dict = {frozenset(): 0.0}

    for size in range(1, k + 1):
        for subset in combinations(range(k), size):
            fs = frozenset(subset)
            X_sub = Xp_resid[:, list(subset)]
            r2_cache[fs] = _r2_ols(y_resid, X_sub)

    # ---- Compute general dominance weights ----
    # For each predictor j:
    #   For each subset size s in {0, ..., k-1}:
    #     average additional R² of adding j to all subsets of size s not containing j
    #   General dominance weight = average of the size-level averages

    weights = np.zeros(k)

    for j in range(k):
        size_averages = []
        for s in range(k):   # subset sizes from 0 to k-1
            increments = []
            for subset in combinations([i for i in range(k) if i != j], s):
                fs_without = frozenset(subset)
                fs_with    = frozenset(subset) | {j}
                increments.append(r2_cache[fs_with] - r2_cache[fs_without])
            if increments:
                size_averages.append(np.mean(increments))
        weights[j] = np.mean(size_averages)

    total_weight = weights.sum()
    pct = (weights / total_weight * 100) if total_weight > 1e-10 else np.zeros(k)

    da_df = pd.DataFrame({
        "predictor":        [PREDICTOR_LABELS.get(p, p) for p in predictor_names],
        "predictor_code":   predictor_names,
        "dominance_weight": weights,
        "pct_of_r2":        pct,
    }).sort_values("dominance_weight", ascending=False).reset_index(drop=True)

    return da_df


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def plot_scatterplots(df: pd.DataFrame, health: str, predictors: list,
                      sp_pretty: str, outpath: Path) -> None:
    """One scatterplot per predictor vs the health metric, with OLS regression line."""
    ncols = 2
    nrows = int(np.ceil(len(predictors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()
    h_label = HEALTH_LABELS.get(health, health)

    for j, pred in enumerate(predictors):
        ax = axes[j]
        x = df[pred].values
        y = df[health].values
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]

        ax.scatter(xm, ym, s=14, alpha=0.35, color="steelblue", linewidths=0)

        if len(xm) > 2:
            slope, intercept, r, p_val, _ = stats.linregress(xm, ym)
            xfit = np.linspace(xm.min(), xm.max(), 200)
            ax.plot(xfit, intercept + slope * xfit, color="firebrick", linewidth=1.6)
            sig = ("***" if p_val < 0.001 else
                   "**"  if p_val < 0.01  else
                   "*"   if p_val < 0.05  else "ns")
            ax.set_title(f"r = {r:.3f}  {sig}", fontsize=9)

        ax.set_xlabel(PREDICTOR_LABELS.get(pred, pred), fontsize=9)
        ax.set_ylabel(h_label, fontsize=9)

    for j in range(len(predictors), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{sp_pretty} – {h_label}\nScatterplots: predictors vs outcome", fontsize=11)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_residual_diagnostics(resid: np.ndarray, fitted: np.ndarray,
                               health: str, sp_pretty: str, outpath: Path) -> dict:
    """Residuals vs Fitted | Histogram | Q-Q | Scale-Location."""
    sw_stat, sw_p = (stats.shapiro(resid) if len(resid) <= 5000
                     else (np.nan, np.nan))

    h_label = HEALTH_LABELS.get(health, health)
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(fitted, resid, s=14, alpha=0.35, color="steelblue", linewidths=0)
    ax1.axhline(0, color="firebrick", linewidth=1.2, linestyle="--")
    ax1.set_xlabel("Fitted values"); ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(resid, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Residual"); ax2.set_ylabel("Count")
    sw_label = (f"Shapiro-Wilk: W = {sw_stat:.4f}, p = {sw_p:.4f}"
                if np.isfinite(sw_stat) else "Shapiro-Wilk: n > 5000")
    ax2.set_title(f"Residual histogram\n{sw_label}", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax3.scatter(osm, osr, s=14, alpha=0.35, color="steelblue", linewidths=0)
    ax3.plot(osm, slope * np.array(osm) + intercept, color="firebrick", linewidth=1.4)
    ax3.set_xlabel("Theoretical quantiles"); ax3.set_ylabel("Sample quantiles")
    ax3.set_title("Normal Q-Q plot")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(fitted, np.sqrt(np.abs(resid)), s=14, alpha=0.35,
                color="steelblue", linewidths=0)
    ax4.set_xlabel("Fitted values"); ax4.set_ylabel("√|Residuals|")
    ax4.set_title("Scale-Location")

    fig.suptitle(f"{sp_pretty} – {h_label}\nResidual diagnostics", fontsize=12)
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)

    return {"shapiro_W": sw_stat, "shapiro_p": sw_p}


def plot_vif(vif_df: pd.DataFrame, health: str, sp_pretty: str, outpath: Path) -> None:
    """Horizontal bar chart of VIF values with threshold lines."""
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.55 * len(vif_df)))
    colors = ["firebrick" if v > 10 else ("darkorange" if v > 5 else "steelblue")
              for v in vif_df["VIF"]]
    ax.barh(vif_df["variable"], vif_df["VIF"], color=colors)
    ax.axvline(5,  color="darkorange", linestyle="--", linewidth=1.2, label="VIF = 5")
    ax.axvline(10, color="firebrick",  linestyle="--", linewidth=1.2, label="VIF = 10")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title(f"{sp_pretty} – {HEALTH_LABELS.get(health, health)}\nMulticollinearity (VIF)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_coefficients(result, health: str, sp_pretty: str, outpath: Path) -> None:
    """Forest plot of MLR coefficients with 95% CI."""
    params = result.params.drop("const", errors="ignore")
    conf   = result.conf_int().drop("const", errors="ignore")
    pvals  = result.pvalues.drop("const", errors="ignore")

    labels = [PREDICTOR_LABELS.get(v, HEALTH_LABELS.get(v, v)) for v in params.index]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 0.7 + 0.6 * len(labels)))
    colors = ["firebrick" if p < ALPHA else "steelblue" for p in pvals]

    ax.barh(y, params.values, color=colors, alpha=0.75)
    ax.errorbar(params.values, y,
                xerr=[params.values - conf[0].values,
                      conf[1].values - params.values],
                fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(
        f"{sp_pretty} – {HEALTH_LABELS.get(health, health)}\n"
        f"MLR coefficients  (red = p < {ALPHA})"
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_dominance(da_df: pd.DataFrame, health: str, sp_pretty: str,
                   outpath: Path) -> None:
    """Horizontal stacked bar showing each predictor's dominance weight."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5))

    # ---- Top: stacked bar of dominance weights ----
    ax = axes[0]
    left = 0.0
    for i, row in da_df.iterrows():
        colour = DA_COLOURS[i % len(DA_COLOURS)]
        ax.barh(0, row["dominance_weight"], left=left, color=colour,
                label=f"{row['predictor']} ({row['pct_of_r2']:.1f}%)", height=0.5)
        if row["dominance_weight"] > 0.005:
            ax.text(left + row["dominance_weight"] / 2, 0,
                    f"{row['pct_of_r2']:.1f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += row["dominance_weight"]

    ax.set_xlim(0, left * 1.02)
    ax.set_yticks([])
    ax.set_xlabel("Dominance weight (average additional R²)")
    ax.set_title(
        f"{sp_pretty} – {HEALTH_LABELS.get(health, health)}\n"
        f"Dominance analysis  (total R² from predictors = {left:.3f})"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)

    # ---- Bottom: individual bar per predictor ----
    ax2 = axes[1]
    colours = [DA_COLOURS[i % len(DA_COLOURS)] for i in range(len(da_df))]
    ax2.barh(da_df["predictor"], da_df["dominance_weight"], color=colours)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Dominance weight")
    ax2.set_title("Dominance weights per predictor")

    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# COMBINED SUMMARY PLOTS
# ============================================================

def plot_combined_r2(summary_df: pd.DataFrame, outpath: Path) -> None:
    """Heatmap of adjusted R² per species × health metric."""
    piv = (summary_df
           .drop_duplicates(subset=["species", "health_metric"])
           [["species", "health_metric", "R2_adj"]]
           .pivot(index="species", columns="health_metric", values="R2_adj"))

    fig, ax = plt.subplots(figsize=(10, 0.8 + 0.6 * len(piv)))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Adjusted R²")
    ax.set_title("Adjusted R² – MLR per species and health metric")
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_combined_morans(summary_df: pd.DataFrame, outpath: Path) -> None:
    """Dot plot of Moran's I on MLR residuals per species × health metric."""
    sub = (summary_df
           .drop_duplicates(subset=["species", "health_metric"])
           [["species", "health_metric", "morans_I", "spatial_autocorr"]]
           .dropna())

    species_list = sub["species"].unique()
    health_list  = sub["health_metric"].unique()
    y_ticks = {sp: i for i, sp in enumerate(species_list)}

    fig, axes = plt.subplots(
        1, len(health_list),
        figsize=(3 * len(health_list), 1.5 + 0.55 * len(species_list)),
        sharey=True
    )
    if len(health_list) == 1:
        axes = [axes]

    for ax, hm in zip(axes, health_list):
        hm_sub = sub[sub["health_metric"] == hm]
        colors = ["firebrick" if sig else "steelblue" for sig in hm_sub["spatial_autocorr"]]
        y_pos  = [y_ticks[sp] for sp in hm_sub["species"]]
        ax.scatter(hm_sub["morans_I"].values, y_pos, c=colors, s=70, zorder=3)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Moran's I", fontsize=8)
        ax.set_title(hm, fontsize=8, wrap=True)
        ax.set_yticks(list(y_ticks.values()))
        ax.set_yticklabels(list(y_ticks.keys()), fontsize=8)

    fig.suptitle(
        "Moran's I on MLR residuals  (red = significant spatial autocorrelation)",
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_combined_dominance(da_all: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap of dominance weights (% of R²) across all species × health metrics,
    with one panel per predictor.
    """
    predictors_pretty = da_all["predictor"].unique()
    health_pretty     = da_all["health_metric"].unique()
    species_pretty    = da_all["species"].unique()

    n_pred = len(predictors_pretty)
    fig, axes = plt.subplots(1, n_pred,
                              figsize=(4.5 * n_pred, 1.0 + 0.55 * len(species_pretty)),
                              sharey=True)
    if n_pred == 1:
        axes = [axes]

    for ax, pred in zip(axes, predictors_pretty):
        sub = da_all[da_all["predictor"] == pred].copy()
        piv = sub.pivot_table(
            index="species", columns="health_metric",
            values="pct_of_r2", aggfunc="mean"
        ).reindex(columns=health_pretty)

        im = ax.imshow(piv.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(health_pretty)))
        ax.set_xticklabels(health_pretty, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_title(pred, fontsize=9)

        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=7, color="white" if val > 55 else "black")

        plt.colorbar(im, ax=ax, label="% of R²", shrink=0.7)

    fig.suptitle("Dominance analysis: % of R² per predictor × species × health metric",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_combined_dominance_stacked(da_all: pd.DataFrame, outpath: Path) -> None:
    """
    One stacked bar per health metric, averaged across species,
    showing how R² is partitioned among predictors.
    """
    health_list = list(da_all["health_metric"].unique())
    pred_list   = list(da_all["predictor"].unique())

    agg = (da_all.groupby(["health_metric", "predictor"])["dominance_weight"]
           .mean().reset_index())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(health_list))
    bar_width = 0.55

    bottoms = np.zeros(len(health_list))
    for pi, pred in enumerate(pred_list):
        vals = []
        for hm in health_list:
            row = agg[(agg["health_metric"] == hm) & (agg["predictor"] == pred)]
            vals.append(row["dominance_weight"].values[0] if len(row) else 0.0)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, width=bar_width,
               color=DA_COLOURS[pi % len(DA_COLOURS)], label=pred)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(health_list, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average dominance weight (R²)")
    ax.set_title("Dominance analysis – average R² partition across species")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# CORE  MLR  +  DOMINANCE  FUNCTION
# ============================================================

def run_mlr_for_health(df_sub: pd.DataFrame, health: str,
                        predictors: list, controls: list,
                        sp_pretty: str, out_dir: Path) -> tuple:
    """
    Fit MLR + dominance analysis for one health metric.
    Returns (mlr_rows, da_rows) — lists of dicts for the summary CSVs.
    """
    all_x_cols = predictors + controls
    needed     = [health] + all_x_cols + COORD_COLS

    d = df_sub[needed].dropna().copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    if len(d) < MIN_N:
        print(f"    [skip] {health}: only {len(d)} obs (min {MIN_N})")
        return [], []

    y      = d[health].values.astype(float)
    X_df   = d[all_x_cols].copy()
    X_sm   = sm.add_constant(X_df, has_constant="add")
    n      = len(d)
    h_safe = health.replace("/", "_")

    # ---- Fit full MLR ----
    try:
        result = sm.OLS(y, X_sm).fit()
    except Exception as e:
        print(f"    [error] {health}: {e}")
        return [], []

    fitted = result.fittedvalues.values
    resid  = result.resid.values

    # ---- 1. Scatterplots ----
    plot_scatterplots(d, health, predictors, sp_pretty,
                      out_dir / f"{h_safe}_scatterplots.png")

    # ---- 2. Residual diagnostics ----
    sw = plot_residual_diagnostics(resid, fitted, health, sp_pretty,
                                    out_dir / f"{h_safe}_residual_diagnostics.png")

    # ---- 3. VIF ----
    X_vif    = X_sm.drop(columns=["const"], errors="ignore")
    vif_vals = [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]
    vif_labels = [PREDICTOR_LABELS.get(c, HEALTH_LABELS.get(c, c))
                  for c in X_vif.columns]
    vif_df = pd.DataFrame({"variable": vif_labels, "VIF": vif_vals})
    plot_vif(vif_df, health, sp_pretty, out_dir / f"{h_safe}_VIF.png")

    # ---- 4. Coefficient forest plot ----
    plot_coefficients(result, health, sp_pretty,
                      out_dir / f"{h_safe}_coefficients.png")

    # ---- 5. Moran's I on residuals ----
    xy    = d[COORD_COLS].values.astype(float)
    k_eff = min(K_NEIGHBORS, n - 2)
    W     = knn_weights(xy, k=k_eff)
    I_obs, p_moran = morans_I_perm(resid, W, n_perm=N_PERMUTATIONS, seed=SEED_MORAN)

    # ---- 6. Dominance analysis ----
    X_pred_arr = d[predictors].values.astype(float)
    X_ctrl_arr = d[controls].values.astype(float) if controls else np.empty((n, 0))

    da_df = dominance_analysis(y, X_pred_arr, X_ctrl_arr, predictors)
    da_df["species"]       = sp_pretty
    da_df["health_metric"] = HEALTH_LABELS.get(health, health)
    da_df.to_csv(out_dir / f"{h_safe}_dominance.csv", index=False)
    plot_dominance(da_df, health, sp_pretty, out_dir / f"{h_safe}_dominance.png")

    # ---- Save OLS summary text ----
    with open(out_dir / f"{h_safe}_OLS_summary.txt", "w") as f:
        f.write(result.summary().as_text())
        f.write(f"\n\nMoran's I on residuals: I = {I_obs:.4f}, "
                f"p (permutation, {N_PERMUTATIONS} perms) = {p_moran:.4f}")
        f.write(f"\nShapiro-Wilk: W = {sw['shapiro_W']:.4f}, p = {sw['shapiro_p']:.4f}")
        f.write("\n\nDominance analysis:\n")
        f.write(da_df[["predictor", "dominance_weight", "pct_of_r2"]].to_string(index=False))

    # ---- Build MLR summary rows (one per predictor) ----
    mlr_rows = []
    ci = result.conf_int()
    for pred in predictors:
        if pred not in result.params.index:
            continue
        mlr_rows.append({
            "species":          sp_pretty,
            "health_metric":    HEALTH_LABELS.get(health, health),
            "predictor":        PREDICTOR_LABELS.get(pred, pred),
            "coefficient":      result.params[pred],
            "ci_lower":         ci.loc[pred, 0],
            "ci_upper":         ci.loc[pred, 1],
            "t_stat":           result.tvalues[pred],
            "p_value":          result.pvalues[pred],
            "significant":      result.pvalues[pred] < ALPHA,
            "R2":               result.rsquared,
            "R2_adj":           result.rsquared_adj,
            "n":                n,
            "shapiro_W":        sw["shapiro_W"],
            "shapiro_p":        sw["shapiro_p"],
            "morans_I":         I_obs,
            "morans_p":         p_moran,
            "spatial_autocorr": p_moran < ALPHA if np.isfinite(p_moran) else False,
        })

    # ---- Build dominance summary rows ----
    da_rows = da_df.to_dict("records")

    return mlr_rows, da_rows


# ============================================================
# MAIN
# ============================================================

def main():
    all_mlr_rows = []
    all_da_rows  = []

    for species, csv_path in SPECIES_CSVS.items():
        print(f"\n{'='*60}")
        print(f"  {species}")
        print(f"{'='*60}")

        sp_out     = OUT_ROOT / species
        ensure_dir(sp_out)
        sp_pretty  = SPECIES_LABELS.get(species, species)

        # ---- Load ----
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [skip] Could not read CSV: {e}")
            continue

        required = HEALTH_VARS + PREDICTORS + CONTROL_VARS
        missing  = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [skip] Missing columns: {missing}")
            continue

        # ---- Attach coordinates ----
        try:
            df = attach_xy_from_shapefile(df)
        except Exception as e:
            print(f"  [skip] Could not attach coordinates: {e}")
            continue

        if not all(c in df.columns for c in COORD_COLS):
            print(f"  [skip] Coordinate columns missing after join.")
            continue
        if df[COORD_COLS].isna().all().all():
            print(f"  [skip] All-NaN coordinates (ID mismatch?).")
            continue

        # ---- Clean ----
        df = make_numeric(df, required + COORD_COLS)
        df = apply_filters(df, required + COORD_COLS)
        print(f"  n = {len(df)} trees after filtering")

        # ---- Run per health metric ----
        sp_mlr_rows, sp_da_rows = [], []

        for health in HEALTH_VARS:
            print(f"  → {HEALTH_LABELS.get(health, health)}")
            h_out = sp_out / health
            ensure_dir(h_out)

            mlr_rows, da_rows = run_mlr_for_health(
                df, health, PREDICTORS, CONTROL_VARS, sp_pretty, h_out
            )
            sp_mlr_rows.extend(mlr_rows)
            sp_da_rows.extend(da_rows)

        if not sp_mlr_rows:
            print(f"  [skip] No results for {species}")
            continue

        # Save per-species tables
        pd.DataFrame(sp_mlr_rows).to_csv(
            sp_out / "MLR_results_all_health_metrics.csv", index=False
        )
        if sp_da_rows:
            pd.DataFrame(sp_da_rows).to_csv(
                sp_out / "dominance_all_health_metrics.csv", index=False
            )

        all_mlr_rows.extend(sp_mlr_rows)
        all_da_rows.extend(sp_da_rows)
        print(f"  [ok] Saved to {sp_out}")

    # ---- Combined outputs ----
    if all_mlr_rows:
        all_mlr_df = pd.DataFrame(all_mlr_rows)
        all_mlr_df.to_csv(OUT_ROOT / "ALL_species_MLR_results.csv", index=False)
        plot_combined_r2(all_mlr_df, OUT_ROOT / "ALL_species_R2_heatmap.png")
        plot_combined_morans(all_mlr_df, OUT_ROOT / "ALL_species_MoransI_residuals.png")

    if all_da_rows:
        all_da_df = pd.DataFrame(all_da_rows)
        all_da_df.to_csv(OUT_ROOT / "ALL_species_dominance_results.csv", index=False)
        plot_combined_dominance(all_da_df, OUT_ROOT / "ALL_species_dominance_heatmap.png")
        plot_combined_dominance_stacked(all_da_df, OUT_ROOT / "ALL_species_dominance_stacked.png")

    if all_mlr_rows or all_da_rows:
        print(f"\n[ok] All outputs saved to: {OUT_ROOT}")
    else:
        print("\n[warn] No results produced. Check paths, column names, and MIN_N.")


if __name__ == "__main__":
    main()