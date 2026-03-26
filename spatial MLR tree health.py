#!/usr/bin/env python3
"""
Multi-species MLR workflow: OLS vs Spatial Error Model (SEM) vs Spatial Lag Model (SLM)
with Dominance Analysis and full model comparison.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW THE THREE MODELS DIFFER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLS (baseline):    y  =  Xβ + ε
  Standard MLR. Assumes errors are independent. Moran's I on residuals
  tells you whether this assumption is violated.

SEM (spatial error model):    y  =  Xβ + u,   u = λWu + ε
  The error term itself is spatially autocorrelated (parameter λ).
  Interpretation: unmeasured confounders are spatially clustered.
  Coefficient estimates are unbiased but SE/p-values from OLS are wrong.
  Moran's I is computed on TRANSFORMED residuals: e_t = (I−λW)y − (I−λW)Xβ
  These should be near zero if SEM has absorbed the spatial structure.

SLM (spatial lag model):    y  =  ρWy + Xβ + ε
  The outcome at each tree depends on outcomes at neighbouring trees (ρ).
  Interpretation: tree health is spatially contagious (shared management,
  microclimate spread, disease, soil continuity).
  Moran's I is computed on raw residuals: yr − Xβ  where  yr = y − ρWy.

MODEL SELECTION GUIDE
  1. Run OLS → check Moran's I on residuals. If significant: use spatial model.
  2. Lagrange Multiplier (LM) tests on OLS residuals guide which spatial model:
       LM-error significant, LM-lag not → prefer SEM
       LM-lag significant, LM-error not → prefer SLM
       Both significant                 → prefer model with lower robust LM statistic
       Neither significant              → OLS is fine
  3. AIC: lower = better fit. Compare OLS vs SEM vs SLM.
  4. Check Moran's I on SEM/SLM residuals: did the spatial model resolve autocorrelation?

DOMINANCE ANALYSIS NOTE
  Dominance analysis (Budescu 1993) is run on the OLS model, partialling out height.
  Decomposing R² across spatial models is not standard practice because SEM/SLM
  log-likelihood does not partition into predictor-wise contributions in the same way.
  The dominance weights therefore reflect predictor importance under the non-spatial
  linear approximation and should be interpreted accordingly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUTS PER SPECIES × HEALTH METRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {health}_scatterplots.png          — predictor vs outcome scatter
  {health}_VIF.png                   — multicollinearity check (OLS)
  {health}_OLS_residuals.png         — residual diagnostics for OLS
  {health}_SEM_residuals.png         — residual diagnostics for SEM
  {health}_SLM_residuals.png         — residual diagnostics for SLM
  {health}_coefficients_comparison.png — OLS vs SEM vs SLM side-by-side
  {health}_moran_comparison.png      — Moran's I before/after spatial correction
  {health}_AIC_comparison.png        — |ΔAIC| vs OLS for SEM and SLM
  {health}_dominance.png/.csv        — dominance analysis (OLS-based)
  {health}_model_summary.txt         — full text output for all three models

COMBINED ACROSS ALL SPECIES
  ALL_species_model_comparison.csv   — one row per species × metric × predictor
  ALL_R2_OLS_heatmap.png
  ALL_AIC_comparison.png             — |ΔAIC| grouped bar chart per health metric
  ALL_delta_AIC_heatmap.png          — NEW: heatmap of |ΔAIC| with winning model
  ALL_Morans_comparison.png
  ALL_species_dominance.csv
  ALL_dominance_stacked.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Anselin, L. (1988). Spatial Econometrics. Kluwer.
  Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996).
      Simple diagnostic tests for spatial dependence. Regional Science
      and Urban Economics, 26(1), 77–104.
  Budescu, D. V. (1993). Dominance analysis. Psychological Bulletin, 114(3), 542–551.
  Azen, R., & Budescu, D. V. (2003). Dominance analysis approach.
      Psychological Methods, 8(2), 129–148.
  Grömping, U. (2006). Relative importance for linear regression in R.
      Journal of Statistical Software, 17(1), 1–27.
  Legendre, P. (1993). Spatial autocorrelation: trouble or new paradigm?
      Ecology, 74(6), 1659–1673.
  O'Brien, R. M. (2007). A caution regarding rules of thumb for VIF.
      Quality & Quantity, 41(5), 673–690.
  Rüttenauer, T. (2022). Spatial regression models: a systematic comparison.
      Sociological Methods & Research, 51(2), 764–803.
  Fusco, E., & Vidoli, F. (2022). Evaluating the performance of AIC and BIC
      for selecting spatial econometric models. Journal of Spatial Econometrics, 3(1).

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
from scipy import stats, optimize

import geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.neighbors import NearestNeighbors


# ╔══════════════════════════════════════════════════════════╗
# ║                   USER SETTINGS                         ║
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
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Model variables
CONTROL_VARS = ["height"]
HEALTH_VARS  = ["ndvi_peak", "sos_doy", "los_days", "amplitude", "auc_above_base_full"]
PREDICTORS   = ["imperv_100m", "poll_bc_anmean", "lst_temp_r100_y", "insolation9"]

COORD_COLS = ["x", "y"]

# Shapefile join
SHAPEFILE_PATH             = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/crown_shapes_final_CRS.shp"
CSV_ID_COL                 = "tree_id"
SHP_ID_COL                 = "crown_id"
USE_CENTROID_IF_NOT_POINTS = True
TARGET_CRS                 = None

# Analysis settings
MIN_N          = 30
ALPHA          = 0.05
K_NEIGHBORS    = 8
N_PERMUTATIONS = 999
SEED_MORAN     = 42
MAX_PREDICTORS_DA = 10
FIG_DPI        = 200

BIOLOGICAL_FILTERS = {
    "los_days":            lambda x: (x >= 60)   & (x <= 365),
    "sos_doy":             lambda x: (x >= 1)    & (x <= 250),
    "ndvi_peak":           lambda x: (x >= -0.1) & (x <= 1.0),
    "amplitude":           lambda x: (x >= -0.1) & (x <= 1.0),
    "auc_above_base_full": lambda x: x > -1e9,
    "height":              lambda x: x > 1,
    "poll_bc_anmean":      lambda x: x > 0,
}

DA_COLOURS    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                 "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
MODEL_COLOURS = {"OLS": "#4C72B0", "SEM": "#DD8452", "SLM": "#55A868"}


# ╔══════════════════════════════════════════════════════════╗
# ║                     UTILITY                             ║
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
    gdf_xy[xn] = geom.x; gdf_xy[yn] = geom.y
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


# ╔══════════════════════════════════════════════════════════╗
# ║               SPATIAL WEIGHTS MATRIX                    ║
# ╚══════════════════════════════════════════════════════════╝

def knn_weights(xy: np.ndarray, k: int) -> np.ndarray:
    """Row-standardised KNN spatial weights matrix (W)."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nbrs.kneighbors(xy)
    idx = idx[:, 1:]
    n = xy.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, idx[i]] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    return np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)


# ╔══════════════════════════════════════════════════════════╗
# ║                    MORAN'S I                            ║
# ╚══════════════════════════════════════════════════════════╝

def morans_I(resid: np.ndarray, W: np.ndarray) -> float:
    z = resid - np.nanmean(resid)
    n = z.size; den = z @ z
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


# ╔══════════════════════════════════════════════════════════╗
# ║         SPATIAL MODELS  (maximum likelihood)            ║
# ╚══════════════════════════════════════════════════════════╝

def fit_sem(y: np.ndarray, X: np.ndarray, W: np.ndarray) -> dict:
    """
    Spatial Error Model: y = Xβ + u,  u = λWu + ε
    ML estimation via grid search over λ.
    Moran's I on TRANSFORMED residuals e_t = (I−λW)y − (I−λW)Xβ.
    """
    n = len(y)
    eigs = np.linalg.eigvalsh(W)
    lam_lo = 1.0 / eigs.min() + 1e-6
    lam_hi = 1.0 / eigs.max() - 1e-6

    def _neg_ll(lam):
        A  = np.eye(n) - lam * W
        Ay = A @ y; AX = A @ X
        try:
            beta = np.linalg.solve(AX.T @ AX, AX.T @ Ay)
        except np.linalg.LinAlgError:
            return 1e10
        e  = Ay - AX @ beta
        s2 = (e @ e) / n
        if s2 <= 0:
            return 1e10
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return 1e10
        return -(logdet - n / 2 * np.log(s2) - n / 2)

    res = optimize.minimize_scalar(
        _neg_ll, bounds=(lam_lo, lam_hi),
        method="bounded", options={"xatol": 1e-8}
    )
    lam = res.x
    A   = np.eye(n) - lam * W
    Ay  = A @ y; AX = A @ X
    beta = np.linalg.solve(AX.T @ AX, AX.T @ Ay)
    e_t  = Ay - AX @ beta
    s2   = (e_t @ e_t) / n
    se   = np.sqrt(np.diag(s2 * np.linalg.inv(AX.T @ AX)))

    k   = X.shape[1] + 1
    ll  = -res.fun
    aic = -2 * ll + 2 * k

    return {
        "beta":    beta,
        "se":      se,
        "lambda":  lam,
        "resid":   e_t,
        "sigma2":  s2,
        "log_lik": ll,
        "aic":     aic,
        "n":       n,
    }


def fit_slm(y: np.ndarray, X: np.ndarray, W: np.ndarray) -> dict:
    """
    Spatial Lag Model: y = ρWy + Xβ + ε
    ML estimation via grid search over ρ.
    Moran's I on raw residuals yr − Xβ where yr = y − ρWy.
    """
    n  = len(y)
    Wy = W @ y
    eigs    = np.linalg.eigvalsh(W)
    rho_lo  = 1.0 / eigs.min() + 1e-6
    rho_hi  = 1.0 / eigs.max() - 1e-6

    def _neg_ll(rho):
        yr = y - rho * Wy
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ yr)
        except np.linalg.LinAlgError:
            return 1e10
        e  = yr - X @ beta
        s2 = (e @ e) / n
        if s2 <= 0:
            return 1e10
        sign, logdet = np.linalg.slogdet(np.eye(n) - rho * W)
        if sign <= 0:
            return 1e10
        return -(logdet - n / 2 * np.log(s2) - n / 2)

    res = optimize.minimize_scalar(
        _neg_ll, bounds=(rho_lo, rho_hi),
        method="bounded", options={"xatol": 1e-8}
    )
    rho  = res.x
    yr   = y - rho * Wy
    beta = np.linalg.solve(X.T @ X, X.T @ yr)
    e    = yr - X @ beta
    s2   = (e @ e) / n
    se   = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))

    k   = X.shape[1] + 1
    ll  = -res.fun
    aic = -2 * ll + 2 * k

    return {
        "beta":    beta,
        "se":      se,
        "rho":     rho,
        "resid":   e,
        "sigma2":  s2,
        "log_lik": ll,
        "aic":     aic,
        "n":       n,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║           LAGRANGE MULTIPLIER TESTS                     ║
# ╚══════════════════════════════════════════════════════════╝

def lm_tests(resid_ols: np.ndarray, X: np.ndarray, W: np.ndarray) -> dict:
    """
    Lagrange Multiplier tests for spatial lag and spatial error
    (Anselin et al. 1996). Computed on OLS residuals.
    """
    n  = len(resid_ols)
    e  = resid_ols
    s2 = (e @ e) / n
    We = W @ e
    T  = np.trace(W.T @ W + W @ W)

    lm_err = (e @ We / s2) ** 2 / T
    p_err  = 1 - stats.chi2.cdf(lm_err, df=1)

    WX = W @ X
    XtX_inv = np.linalg.inv(X.T @ X)
    denom_lag = T + np.trace(WX @ XtX_inv @ WX.T) / s2
    lm_lag = (e @ W @ e / s2) ** 2 / denom_lag
    p_lag  = 1 - stats.chi2.cdf(lm_lag, df=1)

    return {
        "LM_error":   lm_err, "p_LM_error": p_err,
        "LM_lag":     lm_lag, "p_LM_lag":   p_lag,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║              DOMINANCE ANALYSIS (OLS)                   ║
# ╚══════════════════════════════════════════════════════════╝

def _r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    try:
        Xc  = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()
        return max(res.rsquared, 0.0)
    except Exception:
        return 0.0

def dominance_analysis(y: np.ndarray, X_predictors: np.ndarray,
                        X_controls: np.ndarray, predictor_names: list) -> pd.DataFrame:
    """
    General dominance weights (Budescu 1993) after partialling out controls.
    """
    k = len(predictor_names)
    if k > MAX_PREDICTORS_DA:
        raise ValueError(f"Too many predictors for dominance analysis ({k}).")

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
# ║                   DIAGNOSTIC PLOTS                      ║
# ╚══════════════════════════════════════════════════════════╝

def plot_scatterplots(df, health, predictors, sp_pretty, outpath):
    ncols = 2; nrows = int(np.ceil(len(predictors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
    axes = np.array(axes).flatten()
    h_label = HEALTH_LABELS.get(health, health)
    for j, pred in enumerate(predictors):
        ax = axes[j]
        x = df[pred].values; y = df[health].values
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]
        ax.scatter(xm, ym, s=14, alpha=0.35, color="steelblue", linewidths=0)
        if len(xm) > 2:
            slope, intercept, r, pv, _ = stats.linregress(xm, ym)
            xf = np.linspace(xm.min(), xm.max(), 200)
            ax.plot(xf, intercept + slope*xf, color="firebrick", lw=1.6)
            sig = "***" if pv<.001 else "**" if pv<.01 else "*" if pv<.05 else "ns"
            ax.set_title(f"r={r:.3f}  {sig}", fontsize=9)
        ax.set_xlabel(PREDICTOR_LABELS.get(pred, pred), fontsize=9)
        ax.set_ylabel(h_label, fontsize=9)
    for j in range(len(predictors), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{sp_pretty} – {h_label}\nScatterplots: predictors vs outcome", fontsize=11)
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)


def plot_residual_diagnostics(resid, fitted, model_label, sp_pretty, health, outpath):
    sw_stat, sw_p = (stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan))
    h_label = HEALTH_LABELS.get(health, health)
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(fitted, resid, s=14, alpha=0.35, color="steelblue", linewidths=0)
    ax1.axhline(0, color="firebrick", lw=1.2, ls="--")
    ax1.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs Fitted")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(resid, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    sw_lbl = (f"Shapiro-Wilk W={sw_stat:.4f}, p={sw_p:.4f}"
              if np.isfinite(sw_stat) else "Shapiro-Wilk: n>5000")
    ax2.set(xlabel="Residual", ylabel="Count", title=f"Histogram\n{sw_lbl}")

    ax3 = fig.add_subplot(gs[1, 0])
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax3.scatter(osm, osr, s=14, alpha=0.35, color="steelblue", linewidths=0)
    ax3.plot(osm, slope*np.array(osm)+intercept, color="firebrick", lw=1.4)
    ax3.set(xlabel="Theoretical quantiles", ylabel="Sample quantiles", title="Normal Q-Q")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(fitted, np.sqrt(np.abs(resid)), s=14, alpha=0.35,
                color="steelblue", linewidths=0)
    ax4.set(xlabel="Fitted values", ylabel="√|Residuals|", title="Scale-Location")

    note = ("Note: residuals are TRANSFORMED (I−λW)e for SEM"
            if model_label == "SEM" else "")
    fig.suptitle(f"{sp_pretty} – {h_label}  [{model_label}]\nResidual diagnostics\n{note}",
                 fontsize=10)
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)
    return {"shapiro_W": sw_stat, "shapiro_p": sw_p}


def plot_vif(vif_df, health, sp_pretty, outpath):
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.55*len(vif_df)))
    colors = ["firebrick" if v>10 else "darkorange" if v>5 else "steelblue"
              for v in vif_df["VIF"]]
    ax.barh(vif_df["variable"], vif_df["VIF"], color=colors)
    ax.axvline(5,  color="darkorange", ls="--", lw=1.2, label="VIF=5")
    ax.axvline(10, color="firebrick",  ls="--", lw=1.2, label="VIF=10")
    ax.set(xlabel="Variance Inflation Factor",
           title=f"{sp_pretty} – {HEALTH_LABELS.get(health,health)}\nVIF (OLS)")
    ax.legend(fontsize=8)
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)


def plot_coefficients_comparison(col_names, ols_full, sem, slm,
                                  health, sp_pretty, outpath):
    """Side-by-side forest plot: OLS vs SEM vs SLM for environmental predictors only."""
    pred_cols = [c for c in col_names if c not in ("const", "height")]
    labels    = [PREDICTOR_LABELS.get(c, c) for c in pred_cols]
    n_pred    = len(pred_cols)
    y_pos     = np.arange(n_pred)

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.75*n_pred))
    offsets = {"OLS": -0.22, "SEM": 0.0, "SLM": 0.22}
    h = 0.18

    for pi, col in enumerate(pred_cols):
        beta = ols_full.params.get(col, np.nan)
        se   = ols_full.bse.get(col, np.nan)
        yp   = y_pos[pi] + offsets["OLS"]
        ax.barh(yp, beta, height=h, color=MODEL_COLOURS["OLS"],
                alpha=0.8, label="OLS" if pi == 0 else "_")
        if np.isfinite(beta) and np.isfinite(se):
            ax.errorbar(beta, yp, xerr=1.96*se,
                        fmt="none", color="black", capsize=3, lw=1.0)

    col_idx = {c: i for i, c in enumerate(col_names)}
    for pi, col in enumerate(pred_cols):
        if col not in col_idx: continue
        beta = sem["beta"][col_idx[col]]
        se   = sem["se"][col_idx[col]]
        yp   = y_pos[pi] + offsets["SEM"]
        ax.barh(yp, beta, height=h, color=MODEL_COLOURS["SEM"],
                alpha=0.8, label=f"SEM (λ={sem['lambda']:.3f})" if pi == 0 else "_")
        ax.errorbar(beta, yp, xerr=1.96*se,
                    fmt="none", color="black", capsize=3, lw=1.0)

    for pi, col in enumerate(pred_cols):
        if col not in col_idx: continue
        beta = slm["beta"][col_idx[col]]
        se   = slm["se"][col_idx[col]]
        yp   = y_pos[pi] + offsets["SLM"]
        ax.barh(yp, beta, height=h, color=MODEL_COLOURS["SLM"],
                alpha=0.8, label=f"SLM (ρ={slm['rho']:.3f})" if pi == 0 else "_")
        ax.errorbar(beta, yp, xerr=1.96*se,
                    fmt="none", color="black", capsize=3, lw=1.0)

    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Coefficient (± 1.96 SE)")
    ax.set_title(f"{sp_pretty} – {HEALTH_LABELS.get(health,health)}\n"
                 f"OLS vs SEM vs SLM coefficients")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)


def plot_moran_comparison(moran_ols, moran_sem, moran_slm,
                           health, sp_pretty, outpath):
    """Bar chart: Moran's I on residuals for OLS, SEM, SLM."""
    models = ["OLS", "SEM\n(transformed\nresiduals)", "SLM"]
    keys   = ["OLS", "SEM", "SLM"]
    vals   = [moran_ols[0], moran_sem[0], moran_slm[0]]
    pvals  = [moran_ols[1], moran_sem[1], moran_slm[1]]
    colors = [MODEL_COLOURS[k] for k in keys]
    sig    = ["*" if (np.isfinite(p) and p < ALPHA) else "" for p in pvals]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, s, v in zip(bars, sig, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                (bar.get_height() if v >= 0 else 0) + 0.005,
                s, ha="center", fontsize=14, color="black")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Moran's I on residuals")
    ax.set_title(f"{sp_pretty} – {HEALTH_LABELS.get(health,health)}\n"
                 f"Residual spatial autocorrelation  (* = p < {ALPHA})")
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║              AIC PLOTS  (updated to |ΔAIC|)             ║
# ╚══════════════════════════════════════════════════════════╝

def plot_aic_comparison(ols_aic, sem_aic, slm_aic, health, sp_pretty, outpath):
    """
    Bar chart showing |ΔAIC| = |model_AIC - OLS_AIC| for SEM and SLM.
    OLS anchors at 0 (baseline at the bottom).
    Taller bar = larger improvement over OLS.
    Bold outline marks the best-fitting spatial model.
    """
    abs_sem = abs(sem_aic - ols_aic) if np.isfinite(sem_aic) else np.nan
    abs_slm = abs(slm_aic - ols_aic) if np.isfinite(slm_aic) else np.nan

    models = ["SEM", "SLM"]
    values = [abs_sem, abs_slm]
    colors = [MODEL_COLOURS["SEM"], MODEL_COLOURS["SLM"]]

    # Winner = largest absolute improvement over OLS
    best_idx = (int(np.nanargmax(values))
                if np.any(np.isfinite(values)) else -1)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(models, values, color=colors, alpha=0.85,
                  edgecolor="white", width=0.45)

    # Bold outline on the best spatial model
    if best_idx >= 0:
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.0)

    # Value labels on top of each bar
    finite_vals = [v for v in values if np.isfinite(v)]
    y_max = max(finite_vals) if finite_vals else 1.0
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + y_max * 0.02,
                    f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("|ΔAIC| vs OLS  (larger = bigger improvement over OLS)")
    ax.set_title(
        f"{sp_pretty} – {HEALTH_LABELS.get(health, health)}\n"
        f"Spatial model improvement over OLS  (bold = best)"
    )

    # Interpretation note
    if best_idx >= 0 and np.isfinite(values[best_idx]):
        best_val = values[best_idx]
        strength = ("very strong" if best_val > 500
                    else "strong"    if best_val > 100
                    else "moderate"  if best_val > 10
                    else "marginal")
        ax.text(0.98, 0.97,
                f"Best: {models[best_idx]}  (|ΔAIC|={best_val:.0f}, {strength})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="dimgrey")

    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_dominance(da_df, health, sp_pretty, outpath):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    ax = axes[0]; left = 0.0
    for i, row in da_df.iterrows():
        col = DA_COLOURS[i % len(DA_COLOURS)]
        ax.barh(0, row["dominance_weight"], left=left, color=col,
                label=f"{row['predictor']} ({row['pct_of_r2']:.1f}%)", height=0.5)
        if row["dominance_weight"] > 0.005:
            ax.text(left + row["dominance_weight"]/2, 0,
                    f"{row['pct_of_r2']:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
        left += row["dominance_weight"]
    ax.set_xlim(0, max(left * 1.02, 0.01)); ax.set_yticks([])
    ax.set_xlabel("Dominance weight (avg additional R²)")
    ax.set_title(f"{sp_pretty} – {HEALTH_LABELS.get(health,health)}\n"
                 f"Dominance analysis [OLS-based]  (total R² from predictors = {left:.3f})")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    ax2 = axes[1]
    cols = [DA_COLOURS[i % len(DA_COLOURS)] for i in range(len(da_df))]
    ax2.barh(da_df["predictor"], da_df["dominance_weight"], color=cols)
    ax2.set_xlabel("Dominance weight"); ax2.set_title("Per-predictor dominance weights")
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║              COMBINED SUMMARY PLOTS                     ║
# ╚══════════════════════════════════════════════════════════╝

def plot_combined_r2(summary_df, outpath):
    piv = (summary_df.drop_duplicates(["species","health_metric"])
           [["species","health_metric","R2_adj_OLS"]]
           .pivot(index="species", columns="health_metric", values="R2_adj_OLS"))
    fig, ax = plt.subplots(figsize=(10, 0.8+0.6*len(piv)))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i,j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v>0.6 else "black")
    plt.colorbar(im, ax=ax, label="Adj. R² (OLS)")
    ax.set_title("Adjusted R² (OLS) per species × health metric")
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)


def plot_combined_aic(summary_df, outpath):
    """
    Grouped bar chart per health metric showing |ΔAIC| for SEM and SLM vs OLS.
    One subplot per health metric, one bar-group per species.
    Taller bar = larger improvement over OLS.
    Bold outline marks the winning spatial model per species.
    """
    health_list = list(summary_df["health_metric"].unique())
    n_hm = len(health_list)

    fig, axes = plt.subplots(1, n_hm, figsize=(4.5 * n_hm, 5), sharey=False)
    if n_hm == 1:
        axes = [axes]

    for ax, hm in zip(axes, health_list):
        sub = (summary_df[summary_df["health_metric"] == hm]
               .drop_duplicates("species"))

        x = np.arange(len(sub))
        w = 0.32

        abs_sem = np.abs(sub["AIC_SEM"].values - sub["AIC_OLS"].values)
        abs_slm = np.abs(sub["AIC_SLM"].values - sub["AIC_OLS"].values)

        b_sem = ax.bar(x - w / 2, abs_sem, width=w,
                       color=MODEL_COLOURS["SEM"], alpha=0.85, label="SEM")
        b_slm = ax.bar(x + w / 2, abs_slm, width=w,
                       color=MODEL_COLOURS["SLM"], alpha=0.85, label="SLM")

        # Bold outline on whichever spatial model wins per species
        for i, (ds, dl) in enumerate(zip(abs_sem, abs_slm)):
            if np.isfinite(ds) and np.isfinite(dl):
                if ds >= dl:
                    b_sem[i].set_edgecolor("black")
                    b_sem[i].set_linewidth(1.8)
                else:
                    b_slm[i].set_edgecolor("black")
                    b_slm[i].set_linewidth(1.8)

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_ylim(bottom=0)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["species"].values, rotation=40,
                           ha="right", fontsize=7)
        ax.set_title(hm, fontsize=9)
        ax.set_ylabel("|ΔAIC| vs OLS", fontsize=8)

        if ax is axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle(
        "|ΔAIC| vs OLS: SEM and SLM per species × health metric\n"
        "Taller bar = larger improvement over OLS  |  bold outline = best spatial model",
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_delta_aic_summary(summary_df, outpath):
    """
    NEW: Heatmap showing max(|ΔAIC_SEM|, |ΔAIC_SLM|) per species × health metric,
    with the winning model (SEM / SLM) annotated in each cell.
    One-page overview of where spatial modelling helps most.
    """
    sub = summary_df.drop_duplicates(["species", "health_metric"]).copy()
    sub["abs_delta_sem"] = np.abs(sub["AIC_SEM"] - sub["AIC_OLS"])
    sub["abs_delta_slm"] = np.abs(sub["AIC_SLM"] - sub["AIC_OLS"])
    sub["best_delta"]    = sub[["abs_delta_sem", "abs_delta_slm"]].max(axis=1)
    sub["winner"]        = np.where(
        sub["abs_delta_sem"] >= sub["abs_delta_slm"], "SEM", "SLM"
    )

    species_list = list(sub["species"].unique())
    health_list  = list(sub["health_metric"].unique())

    piv_delta  = sub.pivot(index="species", columns="health_metric",
                            values="best_delta").reindex(
                                index=species_list, columns=health_list)
    piv_winner = sub.pivot(index="species", columns="health_metric",
                            values="winner").reindex(
                                index=species_list, columns=health_list)

    fig, ax = plt.subplots(figsize=(2.2 * len(health_list),
                                     0.9 + 0.65 * len(species_list)))

    # Cap colorscale at 95th percentile so outliers don't wash out the rest
    vmax = np.nanpercentile(piv_delta.values.astype(float), 95)
    im   = ax.imshow(piv_delta.values.astype(float),
                     aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

    for i in range(len(species_list)):
        for j in range(len(health_list)):
            val    = piv_delta.iloc[i, j]
            winner = piv_winner.iloc[i, j]
            if np.isfinite(float(val)):
                txt_color = "white" if float(val) > vmax * 0.6 else "black"
                ax.text(j, i,
                        f"{float(val):.0f}\n{winner}",
                        ha="center", va="center",
                        fontsize=8, color=txt_color,
                        fontweight="bold")

    ax.set_xticks(range(len(health_list)))
    ax.set_xticklabels(health_list, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(species_list)))
    ax.set_yticklabels(species_list, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("|ΔAIC| vs OLS  (colour capped at 95th percentile)", fontsize=8)

    ax.set_title(
        "Spatial model improvement over OLS\n"
        "|ΔAIC| with winning model (SEM / SLM) per species × health metric",
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_combined_moran(summary_df, outpath):
    health_list  = list(summary_df["health_metric"].unique())
    species_list = list(summary_df["species"].unique())
    y_ticks = {sp: i for i, sp in enumerate(species_list)}
    markers = {"OLS": "o", "SEM": "s", "SLM": "^"}

    fig, axes = plt.subplots(1, len(health_list),
                              figsize=(3.5*len(health_list), 1.5+0.6*len(species_list)),
                              sharey=True)
    if len(health_list) == 1: axes = [axes]

    for ax, hm in zip(axes, health_list):
        sub = summary_df[summary_df["health_metric"]==hm].drop_duplicates("species")
        for model in ["OLS", "SEM", "SLM"]:
            i_col = f"Morans_I_{model}"; p_col = f"Morans_p_{model}"
            yp    = [y_ticks[sp] for sp in sub["species"]]
            sig_c = [MODEL_COLOURS[model] if (np.isfinite(p) and p<ALPHA)
                     else "lightgrey" for p in sub[p_col].values]
            ax.scatter(sub[i_col].values, yp, c=sig_c, s=60,
                       marker=markers[model], label=model, zorder=3, alpha=0.9)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Moran's I", fontsize=8); ax.set_title(hm, fontsize=8)
        ax.set_yticks(list(y_ticks.values()))
        ax.set_yticklabels(list(y_ticks.keys()), fontsize=8)
        if ax is axes[0]: ax.legend(fontsize=7)

    fig.suptitle("Moran's I on residuals: OLS vs SEM vs SLM\n"
                 "(coloured=significant, grey=n.s.  |  SEM uses transformed residuals)",
                 fontsize=10)
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)


def plot_combined_dominance_stacked(da_all, outpath):
    health_list = list(da_all["health_metric"].unique())
    pred_list   = list(da_all["predictor"].unique())
    agg = da_all.groupby(["health_metric","predictor"])["dominance_weight"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(health_list)); bottoms = np.zeros(len(health_list))
    for pi, pred in enumerate(pred_list):
        vals = np.array([
            agg.loc[(agg.health_metric==hm)&(agg.predictor==pred),"dominance_weight"].values[0]
            if len(agg.loc[(agg.health_metric==hm)&(agg.predictor==pred)]) else 0.0
            for hm in health_list
        ])
        ax.bar(x, vals, bottom=bottoms, width=0.55,
               color=DA_COLOURS[pi % len(DA_COLOURS)], label=pred)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(health_list, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg dominance weight (R²)  [OLS-based]")
    ax.set_title("Dominance analysis – average R² partition across species")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    plt.tight_layout(); fig.savefig(outpath, dpi=FIG_DPI); plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║          CORE PER-METRIC FUNCTION                       ║
# ╚══════════════════════════════════════════════════════════╝

def run_all_models(df_sub, health, predictors, controls, sp_pretty, out_dir):
    """
    Fit OLS + SEM + SLM for one species × health metric combination.
    Returns (summary_rows, da_rows).
    """
    all_x_cols = predictors + controls
    needed     = [health] + all_x_cols + COORD_COLS

    d = df_sub[needed].dropna().copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    if len(d) < MIN_N:
        print(f"    [skip] {health}: n={len(d)} < {MIN_N}")
        return [], []

    n      = len(d)
    h_safe = health.replace("/", "_")
    h_lbl  = HEALTH_LABELS.get(health, health)

    y  = d[health].values.astype(float)
    xy = d[COORD_COLS].values.astype(float)
    W  = knn_weights(xy, k=min(K_NEIGHBORS, n-2))

    X_df = d[all_x_cols].copy()
    X_sm = sm.add_constant(X_df, has_constant="add")
    X    = X_sm.values.astype(float)
    col_names = list(X_sm.columns)

    # ── Scatterplots ──────────────────────────────────────────
    plot_scatterplots(d, health, predictors, sp_pretty,
                      out_dir / f"{h_safe}_scatterplots.png")

    # ── VIF (OLS design matrix) ───────────────────────────────
    X_vif = X_sm.drop(columns=["const"], errors="ignore")
    vif_df = pd.DataFrame({
        "variable": [PREDICTOR_LABELS.get(c, c) for c in X_vif.columns],
        "VIF":      [variance_inflation_factor(X_vif.values, i)
                     for i in range(X_vif.shape[1])],
    })
    plot_vif(vif_df, health, sp_pretty, out_dir / f"{h_safe}_VIF.png")

    # ── OLS ───────────────────────────────────────────────────
    ols_full  = sm.OLS(y, X_sm).fit()
    ols_resid = ols_full.resid.values
    ols_fit   = ols_full.fittedvalues.values
    sw_ols    = plot_residual_diagnostics(ols_resid, ols_fit,
                                          "OLS", sp_pretty, health,
                                          out_dir / f"{h_safe}_OLS_residuals.png")
    I_ols, p_ols = morans_I_perm(ols_resid, W, N_PERMUTATIONS, SEED_MORAN)
    lm = lm_tests(ols_resid, X, W)

    if lm["p_LM_error"] < ALPHA and lm["p_LM_lag"] >= ALPHA:
        lm_recommendation = "SEM"
    elif lm["p_LM_lag"] < ALPHA and lm["p_LM_error"] >= ALPHA:
        lm_recommendation = "SLM"
    elif lm["p_LM_error"] < ALPHA and lm["p_LM_lag"] < ALPHA:
        lm_recommendation = "SEM or SLM (both significant — compare robust LM or AIC)"
    else:
        lm_recommendation = "OLS (neither LM test significant)"

    # ── SEM ───────────────────────────────────────────────────
    sem = None; sw_sem = {}; I_sem = p_sem = np.nan
    try:
        sem     = fit_sem(y, X, W)
        fit_sem_vals = X @ sem["beta"]
        sw_sem  = plot_residual_diagnostics(sem["resid"], fit_sem_vals,
                                             "SEM", sp_pretty, health,
                                             out_dir / f"{h_safe}_SEM_residuals.png")
        I_sem, p_sem = morans_I_perm(sem["resid"], W, N_PERMUTATIONS, SEED_MORAN)
    except Exception as e:
        print(f"    [SEM error] {e}")

    # ── SLM ───────────────────────────────────────────────────
    slm = None; sw_slm = {}; I_slm = p_slm = np.nan
    try:
        slm    = fit_slm(y, X, W)
        fit_slm_vals = X @ slm["beta"]
        sw_slm = plot_residual_diagnostics(slm["resid"], fit_slm_vals,
                                            "SLM", sp_pretty, health,
                                            out_dir / f"{h_safe}_SLM_residuals.png")
        I_slm, p_slm = morans_I_perm(slm["resid"], W, N_PERMUTATIONS, SEED_MORAN)
    except Exception as e:
        print(f"    [SLM error] {e}")

    # ── Comparison plots ──────────────────────────────────────
    if sem is not None and slm is not None:
        plot_coefficients_comparison(col_names, ols_full, sem, slm,
                                      health, sp_pretty,
                                      out_dir / f"{h_safe}_coefficients_comparison.png")
        plot_moran_comparison((I_ols, p_ols), (I_sem, p_sem), (I_slm, p_slm),
                               health, sp_pretty,
                               out_dir / f"{h_safe}_moran_comparison.png")
        plot_aic_comparison(ols_full.aic,
                             sem["aic"] if sem else np.nan,
                             slm["aic"] if slm else np.nan,
                             health, sp_pretty,
                             out_dir / f"{h_safe}_AIC_comparison.png")

    # ── Dominance analysis (OLS) ──────────────────────────────
    X_pred_arr = d[predictors].values.astype(float)
    X_ctrl_arr = d[controls].values.astype(float) if controls else np.empty((n, 0))
    da_df = dominance_analysis(y, X_pred_arr, X_ctrl_arr, predictors)
    da_df["species"]       = sp_pretty
    da_df["health_metric"] = h_lbl
    da_df.to_csv(out_dir / f"{h_safe}_dominance.csv", index=False)
    plot_dominance(da_df, health, sp_pretty, out_dir / f"{h_safe}_dominance.png")

    # ── Determine best model ───────────────────────────────────
    aic_vals = {
        "OLS": ols_full.aic,
        "SEM": sem["aic"] if sem else np.inf,
        "SLM": slm["aic"] if slm else np.inf,
    }
    best_aic = min(aic_vals, key=aic_vals.get)

    # ── Text summary ──────────────────────────────────────────
    col_idx = {c: i for i, c in enumerate(col_names)}
    with open(out_dir / f"{h_safe}_model_summary.txt", "w") as f:
        f.write(f"{'='*65}\n{sp_pretty} – {h_lbl}\n{'='*65}\n\n")
        f.write(ols_full.summary().as_text())
        f.write(f"\n\nOLS Moran's I on residuals: I={I_ols:.4f},  p={p_ols:.4f}")
        f.write(f"\n\nLagrange Multiplier tests (on OLS residuals):")
        f.write(f"\n  LM-error: stat={lm['LM_error']:.4f},  p={lm['p_LM_error']:.4f}")
        f.write(f"\n  LM-lag:   stat={lm['LM_lag']:.4f},    p={lm['p_LM_lag']:.4f}")
        f.write(f"\n  → Recommendation: {lm_recommendation}")
        f.write(f"\n\n{'─'*65}\nAIC comparison (lower = better fit):")
        f.write(f"\n  OLS: {ols_full.aic:.2f}")
        if sem:
            f.write(f"\n  SEM: {sem['aic']:.2f}  (λ={sem['lambda']:.4f},  |ΔAIC|={abs(sem['aic']-ols_full.aic):.2f})")
        if slm:
            f.write(f"\n  SLM: {slm['aic']:.2f}  (ρ={slm['rho']:.4f},  |ΔAIC|={abs(slm['aic']-ols_full.aic):.2f})")
        f.write(f"\n  → Best model (AIC): {best_aic}")
        f.write(f"\n\n{'─'*65}\nSpatial model residual autocorrelation (Moran's I):")
        if sem:
            f.write(f"\n  SEM (transformed residuals): I={I_sem:.4f},  p={p_sem:.4f}"
                    f"  {'[significant]' if np.isfinite(p_sem) and p_sem<ALPHA else '[n.s.]'}")
        if slm:
            f.write(f"\n  SLM (raw residuals):         I={I_slm:.4f},  p={p_slm:.4f}"
                    f"  {'[significant]' if np.isfinite(p_slm) and p_slm<ALPHA else '[n.s.]'}")
        f.write(f"\n\n{'─'*65}\nSEM coefficients:")
        if sem:
            for c in col_names:
                if c in col_idx:
                    b = sem["beta"][col_idx[c]]; se_v = sem["se"][col_idx[c]]
                    f.write(f"\n  {PREDICTOR_LABELS.get(c,c):45s}  β={b:+.4f}  SE={se_v:.4f}")
        f.write(f"\n\n{'─'*65}\nSLM coefficients:")
        if slm:
            for c in col_names:
                if c in col_idx:
                    b = slm["beta"][col_idx[c]]; se_v = slm["se"][col_idx[c]]
                    f.write(f"\n  {PREDICTOR_LABELS.get(c,c):45s}  β={b:+.4f}  SE={se_v:.4f}")
        f.write(f"\n\n{'─'*65}\nDominance analysis (OLS-based):\n")
        f.write(da_df[["predictor","dominance_weight","pct_of_r2"]].to_string(index=False))

    # ── Build summary rows ────────────────────────────────────
    def _get(res, key, what="beta"):
        if res is None: return np.nan
        arr = res[what]
        idx = col_idx.get(key)
        return arr[idx] if idx is not None and idx < len(arr) else np.nan

    rows = []
    for pred in predictors:
        p_lbl = PREDICTOR_LABELS.get(pred, pred)
        rows.append({
            "species":           sp_pretty,
            "health_metric":     h_lbl,
            "predictor":         p_lbl,
            # OLS
            "coef_OLS":          ols_full.params.get(pred, np.nan),
            "se_OLS":            ols_full.bse.get(pred, np.nan),
            "p_OLS":             ols_full.pvalues.get(pred, np.nan),
            "sig_OLS":           ols_full.pvalues.get(pred, 1.0) < ALPHA,
            "R2_OLS":            ols_full.rsquared,
            "R2_adj_OLS":        ols_full.rsquared_adj,
            "AIC_OLS":           ols_full.aic,
            "Morans_I_OLS":      I_ols,
            "Morans_p_OLS":      p_ols,
            "LM_error":          lm["LM_error"],
            "p_LM_error":        lm["p_LM_error"],
            "LM_lag":            lm["LM_lag"],
            "p_LM_lag":          lm["p_LM_lag"],
            "LM_recommendation": lm_recommendation,
            # SEM
            "coef_SEM":          _get(sem, pred, "beta"),
            "se_SEM":            _get(sem, pred, "se"),
            "lambda_SEM":        sem["lambda"] if sem else np.nan,
            "AIC_SEM":           sem["aic"]    if sem else np.nan,
            "Morans_I_SEM":      I_sem,
            "Morans_p_SEM":      p_sem,
            # SLM
            "coef_SLM":          _get(slm, pred, "beta"),
            "se_SLM":            _get(slm, pred, "se"),
            "rho_SLM":           slm["rho"]  if slm else np.nan,
            "AIC_SLM":           slm["aic"]  if slm else np.nan,
            "Morans_I_SLM":      I_slm,
            "Morans_p_SLM":      p_slm,
            # Selection
            "best_model_AIC":    best_aic,
            "n":                 n,
            "shapiro_W_OLS":     sw_ols.get("shapiro_W", np.nan),
            "shapiro_p_OLS":     sw_ols.get("shapiro_p", np.nan),
        })

    return rows, da_df.to_dict("records")


# ╔══════════════════════════════════════════════════════════╗
# ║                       MAIN                              ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    all_rows = []
    all_da   = []

    for species, csv_path in SPECIES_CSVS.items():
        print(f"\n{'='*60}\n  {species}\n{'='*60}")
        sp_out    = OUT_ROOT / species
        ensure_dir(sp_out)
        sp_pretty = SPECIES_LABELS.get(species, species)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [skip] Cannot read CSV: {e}"); continue

        required = HEALTH_VARS + PREDICTORS + CONTROL_VARS
        missing  = [c for c in required if c not in df.columns]
        if missing:
            print(f"  [skip] Missing columns: {missing}"); continue

        try:
            df = attach_xy_from_shapefile(df)
        except Exception as e:
            print(f"  [skip] Coordinate join failed: {e}"); continue

        if not all(c in df.columns for c in COORD_COLS):
            print("  [skip] Coordinate columns missing."); continue
        if df[COORD_COLS].isna().all().all():
            print("  [skip] All-NaN coordinates (ID mismatch?)."); continue

        df = make_numeric(df, required + COORD_COLS)
        df = apply_filters(df, required + COORD_COLS)
        print(f"  n = {len(df)} trees after filtering")

        sp_rows, sp_da = [], []
        for health in HEALTH_VARS:
            print(f"  → {HEALTH_LABELS.get(health, health)}")
            h_out = sp_out / health
            ensure_dir(h_out)
            rows, da_rows = run_all_models(
                df, health, PREDICTORS, CONTROL_VARS, sp_pretty, h_out
            )
            sp_rows.extend(rows); sp_da.extend(da_rows)

        if not sp_rows:
            print("  [skip] No results produced."); continue

        pd.DataFrame(sp_rows).to_csv(sp_out / "model_comparison_all_metrics.csv", index=False)
        if sp_da:
            pd.DataFrame(sp_da).to_csv(sp_out / "dominance_all_metrics.csv", index=False)

        all_rows.extend(sp_rows); all_da.extend(sp_da)
        print(f"  [ok] → {sp_out}")

    # Combined outputs
    if all_rows:
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(OUT_ROOT / "ALL_species_model_comparison.csv", index=False)
        plot_combined_r2(all_df, OUT_ROOT / "ALL_R2_OLS_heatmap.png")
        plot_combined_aic(all_df, OUT_ROOT / "ALL_AIC_comparison.png")
        plot_delta_aic_summary(all_df, OUT_ROOT / "ALL_delta_AIC_heatmap.png")
        moran_sum = (all_df[["species","health_metric",
                              "Morans_I_OLS","Morans_p_OLS",
                              "Morans_I_SEM","Morans_p_SEM",
                              "Morans_I_SLM","Morans_p_SLM"]]
                     .drop_duplicates(["species","health_metric"]))
        plot_combined_moran(moran_sum, OUT_ROOT / "ALL_Morans_comparison.png")

    if all_da:
        all_da_df = pd.DataFrame(all_da)
        all_da_df.to_csv(OUT_ROOT / "ALL_species_dominance.csv", index=False)
        plot_combined_dominance_stacked(all_da_df, OUT_ROOT / "ALL_dominance_stacked.png")

    if all_rows or all_da:
        print(f"\n[ok] All outputs saved to: {OUT_ROOT}")
    else:
        print("\n[warn] No results. Check paths, column names, and MIN_N.")


if __name__ == "__main__":
    main()