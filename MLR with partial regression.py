#!/usr/bin/env python3
"""
Multi-species MLR workflow: OLS vs Spatial Error Model (SEM) vs Spatial Lag Model (SLM)
with Dominance Analysis, Partial Regression plots, and full model comparison.

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

PARTIAL REGRESSION NOTE
  Added-variable (partial regression) plots show the unique linear relationship between
  each focal predictor and the outcome, after removing the shared variance with all
  other predictors (including the control, height). For focal predictor xⱼ:
    ê_y  = residuals of y ~ (all predictors except xⱼ)
    ê_xⱼ = residuals of xⱼ ~ (all other predictors)
  Plotting ê_y vs ê_xⱼ yields a slope equal to the OLS partial regression coefficient βⱼ.
  The partial R² = cor(ê_y, ê_xⱼ)² is the fraction of residual variance in y that xⱼ
  explains uniquely, after accounting for all other predictors.

  Interpretation relative to dominance weights:
    - Dominance weight = unique + shared R² contributions (averaged across subsets).
    - Partial R² = unique R² only.
    - A predictor with high dominance weight but low partial R² is largely riding on
      its correlation with other predictors. A predictor with partial R² ≈ dominance
      weight contributes mostly uniquely.
    - When collinearity is high, partial R² values will be much smaller than dominance
      weights, and the scatter in each panel will look noisy despite a real association.
    - The slope (and t-statistic) in the partial regression plot is identical to the
      OLS coefficient and t-statistic — it is just a different visualisation of the
      same β, making leverage points and non-linearities easier to spot.

DAG-MOTIVATED MODEL SPECIFICATIONS
  Three OLS specifications bracket the causal uncertainty around whether LST is a
  mediator (impervious → LST → tree stress) or an independent stressor.

  Spec 1 — Full model (existing):
    y ~ imperv + BC + LST + insolation + height
    Risk: if LST mediates impervious, this over-controls and underestimates imperv effect.

  Spec 2 — Urbanisation-only:
    y ~ imperv + height
    Answers: total effect of impervious cover, assuming LST/BC are downstream (DAG A).
    If imperv beta is much larger here than Spec 1, mediation is likely.

  Spec 3 — Thermal-only:
    y ~ LST + height
    Answers: how much does temperature alone explain, independently of sealing?
    If R2 similar to Spec 1, impervious adds little beyond thermal stress.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUTS PER SPECIES × HEALTH METRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {health}_scatterplots.png            — predictor vs outcome scatter
  {health}_VIF.png                     — multicollinearity check (OLS)
  {health}_OLS_residuals.png           — residual diagnostics for OLS
  {health}_SEM_residuals.png           — residual diagnostics for SEM
  {health}_SLM_residuals.png           — residual diagnostics for SLM
  {health}_coefficients_comparison.png — OLS vs SEM vs SLM side-by-side
  {health}_moran_comparison.png        — Moran's I before/after spatial correction
  {health}_AIC_comparison.png          — |ΔAIC| vs OLS for SEM and SLM
  {health}_dominance.png/.csv          — dominance analysis (OLS-based)
  {health}_partial_regression.png      — added-variable plots (OLS-based)
  {health}_partial_regression.csv      — partial R2, slope, t-stat per predictor
  {health}_dag_sensitivity.png         — NEW: imperv beta and R2 across 3 DAG specs
  {health}_model_summary.txt           — full text output for all models

COMBINED ACROSS ALL SPECIES
  ALL_species_model_comparison.csv     — one row per species x metric x predictor
  ALL_R2_OLS_heatmap.png
  ALL_AIC_comparison.png               — |DAIC| grouped bar chart per health metric
  ALL_delta_AIC_heatmap.png            — heatmap of |DAIC| with winning model
  ALL_Morans_comparison.png
  ALL_species_dominance.csv
  ALL_dominance_stacked.png
  ALL_partial_R2_heatmap.png           — partial R2 heatmap across species x metrics
  ALL_partial_dominance_comparison.png — partial R2 vs dominance weight scatter
  ALL_dag_coef_sensitivity.png         — NEW: imperv beta across 3 specs all species
  ALL_dag_r2_comparison.png            — NEW: R2 of 3 specs per health metric heatmap

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
  Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). Regression Diagnostics.
      Wiley. [added-variable plots, Chapter 2]
  Cook, R. D., & Weisberg, S. (1982). Residuals and Influence in Regression.
      Chapman & Hall. [leverage and partial regression]
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

# DAG-motivated model specifications
# Each entry: (label, predictors_list, description)
# Controls (height) are always added automatically.
DAG_SPECS = [
    (
        "Spec1_Full",
        ["imperv_100m", "poll_bc_anmean", "lst_temp_r100_y", "insolation9"],
        "Full model\n(imperv + BC + LST + insolation)",
    ),
    (
        "Spec2_Urb",
        ["imperv_100m"],
        "Urbanisation-only\n(imperv, total effect — DAG A)",
    ),
    (
        "Spec3_Thermal",
        ["lst_temp_r100_y"],
        "Thermal-only\n(LST, independent stressor — DAG B)",
    ),
]
DAG_COLOURS = {"Spec1_Full": "#4C72B0", "Spec2_Urb": "#DD8452", "Spec3_Thermal": "#55A868"}


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
# ║           PARTIAL REGRESSION (ADDED-VARIABLE)           ║
# ╚══════════════════════════════════════════════════════════╝

def partial_regression_analysis(y: np.ndarray,
                                 X_sm: pd.DataFrame,
                                 predictors: list,
                                 controls: list) -> pd.DataFrame:
    """
    Compute added-variable (partial regression) statistics for each focal predictor.

    For focal predictor xⱼ:
      ê_y   = residuals of OLS(y  ~ intercept + all_other_predictors + controls)
      ê_xⱼ  = residuals of OLS(xⱼ ~ intercept + all_other_predictors + controls)

    Returns a DataFrame with one row per predictor containing:
      resid_y   : ê_y array  (list, for plotting)
      resid_x   : ê_xⱼ array (list, for plotting)
      slope     : ∂y/∂xⱼ   (identical to OLS βⱼ)
      t_stat    : t-statistic of the partial regression slope
      p_value   : two-sided p-value
      partial_r2: cor(ê_y, ê_xⱼ)²
    """
    all_vars = predictors + controls
    rows = []

    for focal in predictors:
        others = [v for v in all_vars if v != focal]

        # ê_y: residualize y on all predictors except focal
        X_others_y = sm.add_constant(X_sm[others].values, has_constant="add")
        ey = sm.OLS(y, X_others_y).fit().resid

        # ê_xⱼ: residualize focal predictor on all others
        xj = X_sm[focal].values.astype(float)
        X_others_x = sm.add_constant(X_sm[others].values, has_constant="add")
        ex = sm.OLS(xj, X_others_x).fit().resid

        # Partial regression: ê_y ~ ê_xⱼ (no intercept needed — both are residuals)
        # Using OLS with constant for numerically stable SE.
        # sm.add_constant on a numpy array returns a numpy array, so use
        # positional indexing [-1] rather than .iloc[-1].
        ex_c = sm.add_constant(ex, has_constant="add")
        pr   = sm.OLS(ey, ex_c).fit()
        slope  = float(pr.params[-1])    # coefficient on ê_xⱼ
        t_stat = float(pr.tvalues[-1])
        p_val  = float(pr.pvalues[-1])

        # Partial R²: squared correlation between the two residual vectors
        partial_r2 = float(np.corrcoef(ey, ex)[0, 1] ** 2)

        rows.append({
            "predictor":   focal,
            "resid_y":     ey,
            "resid_x":     ex,
            "slope":       slope,
            "t_stat":      t_stat,
            "p_value":     p_val,
            "partial_r2":  partial_r2,
        })

    return pd.DataFrame(rows)


def plot_partial_regression(pr_df: pd.DataFrame,
                             ols_full,
                             health: str,
                             sp_pretty: str,
                             outpath: Path) -> None:
    """
    4-panel added-variable plot: one panel per focal predictor.

    Each panel shows:
      - Scatter of ê_y vs ê_xⱼ
      - Partial regression line (slope = OLS βⱼ)
      - 95 % confidence band
      - Annotations: partial R², t-stat, p-value
      - Zero reference lines
      - Observation index for the three most extreme points (Cook-style flags)
    """
    n_pred = len(pr_df)
    ncols  = 2
    nrows  = int(np.ceil(n_pred / ncols))
    h_lbl  = HEALTH_LABELS.get(health, health)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6.5 * ncols, 5.2 * nrows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(pr_df.iterrows()):
        ax     = axes[i]
        ex     = row["resid_x"]
        ey     = row["resid_y"]
        slope  = row["slope"]
        t_val  = row["t_stat"]
        p_val  = row["p_value"]
        pr2    = row["partial_r2"]
        pred   = row["predictor"]
        p_lbl  = PREDICTOR_LABELS.get(pred, pred)

        # ── scatter ──────────────────────────────────────────
        ax.scatter(ex, ey, s=16, alpha=0.35, color="steelblue",
                   linewidths=0, zorder=2)

        # ── regression line + 95 % CI band ───────────────────
        x_sort  = np.sort(ex)
        x_range = np.linspace(ex.min(), ex.max(), 300)
        n       = len(ex)

        # Fitted values and SE of fit
        ex_c    = sm.add_constant(ex, has_constant="add")
        pr_fit  = sm.OLS(ey, ex_c).fit()
        x_pred  = sm.add_constant(x_range, has_constant="add")
        pred_res = pr_fit.get_prediction(x_pred)
        pred_df  = pred_res.summary_frame(alpha=ALPHA)

        ax.plot(x_range, pred_df["mean"], color="firebrick", lw=1.8, zorder=3)
        ax.fill_between(x_range,
                        pred_df["mean_ci_lower"],
                        pred_df["mean_ci_upper"],
                        color="firebrick", alpha=0.12, zorder=1)

        # ── zero reference lines ──────────────────────────────
        ax.axhline(0, color="grey", lw=0.7, ls="--", zorder=0)
        ax.axvline(0, color="grey", lw=0.7, ls="--", zorder=0)

        # ── flag top-3 most influential points (by |ê_y| + |ê_x|) ──
        influence = np.abs(ey) + np.abs(ex)
        top3_idx  = np.argsort(influence)[-3:]
        for idx in top3_idx:
            ax.annotate(str(idx),
                        xy=(ex[idx], ey[idx]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="dimgrey")

        # ── annotation box ────────────────────────────────────
        sig_str = ("***" if p_val < 0.001 else
                   "**"  if p_val < 0.01  else
                   "*"   if p_val < 0.05  else "ns")
        ann = (f"Partial R² = {pr2:.3f}\n"
               f"slope = {slope:.4f}\n"
               f"t = {t_val:.2f},  p = {p_val:.4f}  {sig_str}")
        ax.text(0.03, 0.97, ann,
                transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.75, edgecolor="lightgrey"))

        # ── labels ────────────────────────────────────────────
        ax.set_xlabel(f"ê({p_lbl})\n[residuals after removing all other predictors]",
                      fontsize=8)
        ax.set_ylabel(f"ê({h_lbl})\n[residuals after removing all other predictors]",
                      fontsize=8)
        ax.set_title(p_lbl, fontsize=10, fontweight="bold")

    # hide unused panels
    for j in range(n_pred, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{sp_pretty} – {h_lbl}\n"
        f"Added-variable (partial regression) plots  [OLS-based]\n"
        f"Slope = OLS β;  Partial R² = unique variance explained by each predictor",
        fontsize=10
    )
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║     COMBINED PARTIAL R² PLOTS (across all species)      ║
# ╚══════════════════════════════════════════════════════════╝

def plot_partial_r2_heatmap(pr_all: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap of mean partial R² per predictor × health metric, averaged across species.
    Annotated with the mean value in each cell.
    """
    # Average partial R² across species for each predictor × health combination
    agg = (pr_all
           .groupby(["health_metric", "predictor_code"])["partial_r2"]
           .mean()
           .reset_index())

    pred_codes   = PREDICTORS
    health_codes = [HEALTH_LABELS.get(h, h) for h in HEALTH_VARS]

    mat = np.full((len(pred_codes), len(health_codes)), np.nan)
    for _, r in agg.iterrows():
        pi = pred_codes.index(r["predictor_code"]) if r["predictor_code"] in pred_codes else -1
        try:
            hi = health_codes.index(r["health_metric"])
        except ValueError:
            hi = -1
        if pi >= 0 and hi >= 0:
            mat[pi, hi] = r["partial_r2"]

    pred_labels = [PREDICTOR_LABELS.get(p, p) for p in pred_codes]

    fig, ax = plt.subplots(figsize=(2.5 * len(health_codes), 0.8 + 0.7 * len(pred_codes)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=np.nanmax(mat))

    for i in range(len(pred_codes)):
        for j in range(len(health_codes)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if v > np.nanmax(mat) * 0.6 else "black")

    ax.set_xticks(range(len(health_codes)))
    ax.set_xticklabels(health_codes, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(pred_codes)))
    ax.set_yticklabels(pred_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean partial R²  (averaged across species)")
    ax.set_title(
        "Partial R² per predictor × health metric\n"
        "(mean across species; unique variance explained after partialling out all others)",
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_partial_vs_dominance(pr_all: pd.DataFrame,
                               da_all: pd.DataFrame,
                               outpath: Path) -> None:
    """
    Scatter plot: dominance weight (x) vs partial R² (y) for every
    species × health metric × predictor combination.

    Points above the diagonal: predictor contributes more uniquely than
    its dominance weight suggests (rare when predictors are correlated).
    Points below the diagonal: predictor's dominance weight is inflated
    by shared variance with other predictors.

    A diagonal reference line (y = x) is drawn.
    Points are coloured by predictor for easy identification.
    """
    # Merge on species, health_metric, predictor_code
    pr_merge = pr_all[["species", "health_metric", "predictor_code",
                        "partial_r2"]].copy()

    # da_all uses pretty predictor names; map back to code
    label_to_code = {v: k for k, v in PREDICTOR_LABELS.items()}
    da_merge = da_all[["species", "health_metric",
                        "predictor", "dominance_weight"]].copy()
    da_merge["predictor_code"] = da_merge["predictor"].map(label_to_code)

    merged = pr_merge.merge(
        da_merge[["species", "health_metric", "predictor_code", "dominance_weight"]],
        on=["species", "health_metric", "predictor_code"],
        how="inner"
    )

    if merged.empty:
        print("  [warn] No data to plot for partial R² vs dominance scatter.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for pi, pred_code in enumerate(PREDICTORS):
        sub = merged[merged["predictor_code"] == pred_code]
        if sub.empty:
            continue
        ax.scatter(sub["dominance_weight"], sub["partial_r2"],
                   s=55, alpha=0.75,
                   color=DA_COLOURS[pi % len(DA_COLOURS)],
                   label=PREDICTOR_LABELS.get(pred_code, pred_code),
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Diagonal reference: y = x
    lim_max = max(merged["dominance_weight"].max(),
                  merged["partial_r2"].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max],
            color="grey", lw=1.0, ls="--", zorder=1,
            label="y = x  (unique = total)")

    ax.set_xlabel("Dominance weight  (unique + shared R²)", fontsize=10)
    ax.set_ylabel("Partial R²  (unique R² only)", fontsize=10)
    ax.set_title(
        "Partial R² vs dominance weight\n"
        "Points below diagonal: dominance weight inflated by shared variance\n"
        "Points on diagonal: predictor contributes mostly uniquely",
        fontsize=10
    )
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


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
    abs_sem = abs(sem_aic - ols_aic) if np.isfinite(sem_aic) else np.nan
    abs_slm = abs(slm_aic - ols_aic) if np.isfinite(slm_aic) else np.nan

    models = ["SEM", "SLM"]
    values = [abs_sem, abs_slm]
    colors = [MODEL_COLOURS["SEM"], MODEL_COLOURS["SLM"]]

    best_idx = (int(np.nanargmax(values))
                if np.any(np.isfinite(values)) else -1)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(models, values, color=colors, alpha=0.85,
                  edgecolor="white", width=0.45)

    if best_idx >= 0:
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.0)

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
# ║         DAG-MOTIVATED SPECIFICATION ANALYSIS            ║
# ╚══════════════════════════════════════════════════════════╝

def fit_dag_specs(y: np.ndarray,
                  d: pd.DataFrame,
                  W: np.ndarray,
                  controls: list,
                  sp_pretty: str,
                  h_lbl: str) -> list:
    """
    Fit the three DAG-motivated OLS + SEM specifications and return a list of
    result dicts, one per spec.  SEM is used (not SLM) because SEM wins in the
    large majority of cases and gives a consistent spatial correction across specs.

    Each dict contains:
      spec_key      : e.g. "Spec1_Full"
      spec_label    : human-readable multi-line label
      predictors    : list of predictor column names in this spec
      n             : sample size
      R2            : OLS R² (all predictors + controls)
      R2_adj        : OLS adjusted R²
      coef_imperv   : OLS coefficient for imperv_100m (nan if not in spec)
      se_imperv     : OLS SE for imperv_100m
      p_imperv      : OLS p-value for imperv_100m
      coef_lst      : OLS coefficient for lst_temp_r100_y (nan if not in spec)
      se_lst        : OLS SE for lst_temp_r100_y
      p_lst         : OLS p-value for lst_temp_r100_y
      sem_coef_imperv : SEM coefficient for imperv_100m (nan if not in spec)
      sem_se_imperv   : SEM SE for imperv_100m
      sem_coef_lst    : SEM coefficient for lst_temp_r100_y
      sem_se_lst      : SEM SE for lst_temp_r100_y
      sem_lambda      : estimated lambda
      species         : sp_pretty
      health_metric   : h_lbl
    """
    results = []

    for spec_key, spec_preds, spec_label in DAG_SPECS:
        all_cols = spec_preds + controls
        # Ensure all columns exist in d
        missing = [c for c in all_cols if c not in d.columns]
        if missing:
            print(f"    [DAG {spec_key}] missing columns: {missing}, skipping")
            continue

        sub = d[all_cols + ["_y_"]].dropna() if "_y_" in d.columns else d[all_cols].copy()
        # Use the already-filtered d; y is passed in separately
        X_df  = d[all_cols].copy()
        X_sm  = sm.add_constant(X_df, has_constant="add")
        X_arr = X_sm.values.astype(float)
        col_names = list(X_sm.columns)
        col_idx   = {c: i for i, c in enumerate(col_names)}

        # OLS
        try:
            ols = sm.OLS(y, X_sm).fit()
        except Exception as e:
            print(f"    [DAG {spec_key} OLS error] {e}")
            continue

        def _get_ols(col):
            p  = float(ols.params.get(col, np.nan))
            se = float(ols.bse.get(col,   np.nan))
            pv = float(ols.pvalues.get(col, np.nan))
            return p, se, pv

        co_imp, se_imp, pv_imp = _get_ols("imperv_100m")
        co_lst, se_lst, pv_lst = _get_ols("lst_temp_r100_y")

        # SEM
        sem_co_imp = sem_se_imp = sem_co_lst = sem_se_lst = sem_lam = np.nan
        try:
            sem_res = fit_sem(y, X_arr, W)
            def _get_sem(col):
                idx = col_idx.get(col)
                if idx is None:
                    return np.nan, np.nan
                return float(sem_res["beta"][idx]), float(sem_res["se"][idx])
            sem_co_imp, sem_se_imp = _get_sem("imperv_100m")
            sem_co_lst, sem_se_lst = _get_sem("lst_temp_r100_y")
            sem_lam = float(sem_res["lambda"])
        except Exception as e:
            print(f"    [DAG {spec_key} SEM error] {e}")

        results.append({
            "spec_key":         spec_key,
            "spec_label":       spec_label,
            "predictors":       spec_preds,
            "n":                len(y),
            "R2":               float(ols.rsquared),
            "R2_adj":           float(ols.rsquared_adj),
            "coef_imperv":      co_imp,
            "se_imperv":        se_imp,
            "p_imperv":         pv_imp,
            "coef_lst":         co_lst,
            "se_lst":           se_lst,
            "p_lst":            pv_lst,
            "sem_coef_imperv":  sem_co_imp,
            "sem_se_imperv":    sem_se_imp,
            "sem_coef_lst":     sem_co_lst,
            "sem_se_lst":       sem_se_lst,
            "sem_lambda":       sem_lam,
            "species":          sp_pretty,
            "health_metric":    h_lbl,
        })

    return results


def plot_dag_sensitivity(dag_results: list,
                          health: str,
                          sp_pretty: str,
                          outpath: Path) -> None:
    """
    Per-species × health metric DAG sensitivity figure.

    Two panels:
      Left  — OLS and SEM coefficients for imperv_100m and lst_temp_r100_y
              across the three specifications, with 95% CI bars.
              This is the mediation sensitivity check: if β_imperv drops
              substantially from Spec2 → Spec1, LST is mediating part of the effect.
      Right — Adjusted R² of OLS for each specification.
              Shows how much explanatory power is lost/gained by restricting
              to one predictor family.
    """
    if not dag_results:
        return

    h_lbl     = HEALTH_LABELS.get(health, health)
    spec_keys  = [r["spec_key"]   for r in dag_results]
    spec_labels = [r["spec_label"] for r in dag_results]
    x          = np.arange(len(dag_results))
    w          = 0.28

    fig, (ax_coef, ax_r2) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # ── Left panel: coefficients ──────────────────────────────
    # imperv OLS
    imp_coef = np.array([r["coef_imperv"]     for r in dag_results])
    imp_se   = np.array([r["se_imperv"]        for r in dag_results])
    imp_sem  = np.array([r["sem_coef_imperv"]  for r in dag_results])
    imp_ssem = np.array([r["sem_se_imperv"]    for r in dag_results])

    # lst OLS
    lst_coef = np.array([r["coef_lst"]         for r in dag_results])
    lst_se   = np.array([r["se_lst"]            for r in dag_results])
    lst_sem  = np.array([r["sem_coef_lst"]      for r in dag_results])
    lst_ssem = np.array([r["sem_se_lst"]        for r in dag_results])

    def _plot_coef(ax, xpos, coefs, ses, color, label, marker="o", ls="-"):
        finite = np.isfinite(coefs)
        if not finite.any():
            return
        ax.plot(xpos[finite], coefs[finite],
                marker=marker, color=color, ls=ls, lw=1.6,
                ms=7, label=label, zorder=3)
        for xi, ci, si in zip(xpos[finite], coefs[finite], ses[finite]):
            if np.isfinite(si):
                ax.errorbar(xi, ci, yerr=1.96 * si,
                            fmt="none", color=color, capsize=4, lw=1.2, zorder=2)

    _plot_coef(ax_coef, x,          imp_coef, imp_se,   "#4C72B0",
               "Impervious (OLS)",  marker="o", ls="-")
    _plot_coef(ax_coef, x + 0.02,   imp_sem,  imp_ssem, "#4C72B0",
               "Impervious (SEM)",  marker="s", ls="--")
    _plot_coef(ax_coef, x,          lst_coef, lst_se,   "#DD8452",
               "LST (OLS)",         marker="o", ls="-")
    _plot_coef(ax_coef, x + 0.02,   lst_sem,  lst_ssem, "#DD8452",
               "LST (SEM)",         marker="s", ls="--")

    ax_coef.axhline(0, color="grey", lw=0.8, ls=":")
    ax_coef.set_xticks(x)
    ax_coef.set_xticklabels(spec_labels, fontsize=8)
    ax_coef.set_ylabel("Regression coefficient (± 95% CI)", fontsize=9)
    ax_coef.set_title("Coefficient sensitivity across DAG specifications\n"
                       "Stable β_imperv (Spec2→1): direct effect  |  "
                       "Large drop: LST mediates imperv", fontsize=8)
    ax_coef.legend(fontsize=8, framealpha=0.8)

    # Annotation: % change in imperv beta from Spec2 → Spec1
    spec_keys_list = [r["spec_key"] for r in dag_results]
    if "Spec1_Full" in spec_keys_list and "Spec2_Urb" in spec_keys_list:
        idx1 = spec_keys_list.index("Spec1_Full")
        idx2 = spec_keys_list.index("Spec2_Urb")
        b1   = imp_coef[idx1]; b2 = imp_coef[idx2]
        if np.isfinite(b1) and np.isfinite(b2) and abs(b2) > 1e-10:
            pct_change = (b1 - b2) / abs(b2) * 100
            direction  = "drop" if pct_change < 0 else "increase"
            ax_coef.text(
                0.98, 0.02,
                f"β_imperv Spec2→Spec1: {pct_change:+.1f}% {direction}\n"
                f"({'consistent with mediation' if pct_change < -15 else 'direct effect likely'})",
                transform=ax_coef.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="dimgrey",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="lightgrey")
            )

    # ── Right panel: R² ──────────────────────────────────────
    r2_vals  = np.array([r["R2_adj"] for r in dag_results])
    bar_cols = [DAG_COLOURS.get(k, "steelblue") for k in spec_keys]
    bars = ax_r2.bar(x, r2_vals, color=bar_cols, alpha=0.85,
                     edgecolor="white", width=0.55)
    for bar, v in zip(bars, r2_vals):
        if np.isfinite(v):
            ax_r2.text(bar.get_x() + bar.get_width() / 2,
                       v + max(r2_vals) * 0.02,
                       f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(spec_labels, fontsize=8)
    ax_r2.set_ylabel("Adjusted R² (OLS)", fontsize=9)
    ax_r2.set_ylim(bottom=0)
    ax_r2.set_title("Model fit (adj. R²) per specification\n"
                     "Thermal-only ≈ Full → imperv adds little beyond LST", fontsize=8)

    fig.suptitle(f"{sp_pretty} – {h_lbl}\n"
                 f"DAG specification sensitivity analysis",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def plot_combined_dag_coef(dag_all: pd.DataFrame, outpath: Path) -> None:
    """
    Combined figure: β_imperv (OLS and SEM) across the three DAG specifications
    for every species × health metric, arranged as a grid of small multiples.

    Rows = health metrics, columns = species.
    Within each panel: three x-positions (one per spec), OLS circle + SEM square,
    with 95% CI bars.  A horizontal dashed line at y=0.
    The key visual signal: does β_imperv shrink from Spec2 to Spec1?
    """
    species_list = list(dag_all["species"].unique())
    health_list  = list(dag_all["health_metric"].unique())
    n_sp = len(species_list)
    n_hm = len(health_list)

    fig, axes = plt.subplots(
        n_hm, n_sp,
        figsize=(3.2 * n_sp, 3.0 * n_hm),
        sharex=False, sharey=False,
        squeeze=False
    )

    spec_order  = ["Spec2_Urb", "Spec1_Full"]   # Spec3 has no imperv, skip
    spec_x      = {s: i for i, s in enumerate(spec_order)}
    spec_labels = {"Spec2_Urb": "Spec2\n(urb-only)", "Spec1_Full": "Spec1\n(full)"}

    for ri, hm in enumerate(health_list):
        for ci, sp in enumerate(species_list):
            ax  = axes[ri][ci]
            sub = dag_all[(dag_all["health_metric"] == hm) &
                          (dag_all["species"] == sp) &
                          (dag_all["spec_key"].isin(spec_order))]

            for _, row in sub.iterrows():
                xp = spec_x.get(row["spec_key"], None)
                if xp is None:
                    continue
                # OLS
                if np.isfinite(row["coef_imperv"]):
                    ax.errorbar(xp - 0.1, row["coef_imperv"],
                                yerr=1.96 * row["se_imperv"] if np.isfinite(row["se_imperv"]) else 0,
                                fmt="o", color="#4C72B0", ms=5, capsize=3, lw=1.1,
                                label="OLS" if (ri == 0 and ci == 0 and xp == 0) else "_")
                # SEM
                if np.isfinite(row["sem_coef_imperv"]):
                    ax.errorbar(xp + 0.1, row["sem_coef_imperv"],
                                yerr=1.96 * row["sem_se_imperv"] if np.isfinite(row["sem_se_imperv"]) else 0,
                                fmt="s", color="#DD8452", ms=5, capsize=3, lw=1.1,
                                label="SEM" if (ri == 0 and ci == 0 and xp == 0) else "_")

            ax.axhline(0, color="grey", lw=0.7, ls=":")
            ax.set_xticks(list(spec_x.values()))
            ax.set_xticklabels([spec_labels[s] for s in spec_order], fontsize=6)
            ax.tick_params(axis="y", labelsize=6)

            if ci == 0:
                ax.set_ylabel(hm, fontsize=7, labelpad=2)
            if ri == 0:
                ax.set_title(sp, fontsize=7)

    # Global legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="#4C72B0", ls="none", ms=6, label="OLS"),
        plt.Line2D([0], [0], marker="s", color="#DD8452", ls="none", ms=6, label="SEM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Impervious surface coefficient (β) across DAG specifications\n"
        "Spec2 = total effect (urbanisation-only)  |  Spec1 = partial effect (full model)\n"
        "Large Spec2→Spec1 drop suggests LST/BC mediate part of the impervious effect",
        fontsize=9
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_combined_dag_r2(dag_all: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap grid: adjusted R² of each DAG specification per species × health metric.
    Three sub-heatmaps side by side, one per specification.
    Allows rapid visual comparison of how much explanatory power each spec captures.
    """
    species_list = list(dag_all["species"].unique())
    health_list  = list(dag_all["health_metric"].unique())
    spec_order   = ["Spec1_Full", "Spec2_Urb", "Spec3_Thermal"]
    spec_labels_map = {
        "Spec1_Full":     "Spec1: Full model",
        "Spec2_Urb":      "Spec2: Urbanisation-only",
        "Spec3_Thermal":  "Spec3: Thermal-only",
    }

    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 0.8 + 0.65 * len(species_list)),
                              sharey=True)

    # Determine common vmax across all specs for comparable colour scale
    vmax = dag_all["R2_adj"].max()

    for ax, spec_key in zip(axes, spec_order):
        sub = dag_all[dag_all["spec_key"] == spec_key]
        piv = (sub.pivot(index="species", columns="health_metric", values="R2_adj")
               .reindex(index=species_list, columns=health_list))

        im = ax.imshow(piv.values.astype(float),
                       aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax)

        for i in range(len(species_list)):
            for j in range(len(health_list)):
                v = piv.iloc[i, j]
                if np.isfinite(float(v)):
                    ax.text(j, i, f"{float(v):.3f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if float(v) > vmax * 0.65 else "black")

        ax.set_xticks(range(len(health_list)))
        ax.set_xticklabels(health_list, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(species_list)))
        ax.set_yticklabels(species_list, fontsize=8)
        ax.set_title(spec_labels_map.get(spec_key, spec_key), fontsize=9)

    plt.colorbar(im, ax=axes[-1], label="Adjusted R² (OLS)", shrink=0.8)
    fig.suptitle(
        "Adjusted R² per DAG specification × species × health metric\n"
        "Spec3 ≈ Spec1 → temperature alone captures most variance  |  "
        "Spec2 ≈ Spec1 → impervious alone is sufficient",
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║          CORE PER-METRIC FUNCTION                       ║
# ╚══════════════════════════════════════════════════════════╝

def run_all_models(df_sub, health, predictors, controls, sp_pretty, out_dir):
    """
    Fit OLS + SEM + SLM for one species x health metric combination.
    Also runs the three DAG-motivated specifications.
    Returns (summary_rows, da_rows, pr_rows, dag_rows).
    """
    all_x_cols = predictors + controls
    needed     = [health] + all_x_cols + COORD_COLS

    d = df_sub[needed].dropna().copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    if len(d) < MIN_N:
        print(f"    [skip] {health}: n={len(d)} < {MIN_N}")
        return [], [], [], []

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

    # ── Partial regression (OLS) ──────────────────────────────
    # Uses the same OLS design matrix; controls are included as "others" to partial out.
    pr_df = partial_regression_analysis(
        y        = y,
        X_sm     = X_df,      # DataFrame with predictors + controls (no const)
        predictors = predictors,
        controls   = controls,
    )
    pr_df["species"]       = sp_pretty
    pr_df["health_metric"] = h_lbl

    # Save CSV (drop the array columns for the CSV)
    pr_csv = pr_df.drop(columns=["resid_y", "resid_x"]).copy()
    pr_csv["predictor_label"] = pr_csv["predictor"].map(
        lambda c: PREDICTOR_LABELS.get(c, c))
    pr_csv.to_csv(out_dir / f"{h_safe}_partial_regression.csv", index=False)

    # Plot
    plot_partial_regression(
        pr_df      = pr_df,
        ols_full   = ols_full,
        health     = health,
        sp_pretty  = sp_pretty,
        outpath    = out_dir / f"{h_safe}_partial_regression.png",
    )

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
        f.write(f"\n\n{'─'*65}\nPartial regression (OLS-based):\n")
        f.write(pr_csv[["predictor_label","partial_r2","slope","t_stat","p_value"
                         ]].to_string(index=False))
        f.write(
            "\n\nNote: partial R² = unique variance explained by each predictor after "
            "partialling out all others (including height). "
            "Slope is identical to the OLS coefficient. "
            "Compare partial R² to dominance weight: large gap = high shared variance."
        )

    # ── Build summary rows ────────────────────────────────────
    def _get(res, key, what="beta"):
        if res is None: return np.nan
        arr = res[what]
        idx = col_idx.get(key)
        return arr[idx] if idx is not None and idx < len(arr) else np.nan

    # Build a lookup for partial R² by predictor code
    pr_lookup = {r["predictor"]: r for _, r in pr_df.iterrows()}

    rows = []
    for pred in predictors:
        p_lbl = PREDICTOR_LABELS.get(pred, pred)
        pr_row = pr_lookup.get(pred, {})
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
            # Partial regression
            "partial_r2":        pr_row.get("partial_r2", np.nan),
            "partial_slope":     pr_row.get("slope",      np.nan),
            "partial_t":         pr_row.get("t_stat",     np.nan),
            "partial_p":         pr_row.get("p_value",    np.nan),
        })

    # Collect partial regression rows for combined plots
    pr_export_rows = []
    for _, r in pr_df.iterrows():
        pr_export_rows.append({
            "species":        sp_pretty,
            "health_metric":  h_lbl,
            "predictor_code": r["predictor"],
            "predictor":      PREDICTOR_LABELS.get(r["predictor"], r["predictor"]),
            "partial_r2":     r["partial_r2"],
            "slope":          r["slope"],
            "t_stat":         r["t_stat"],
            "p_value":        r["p_value"],
        })

    # ── DAG specification analysis ────────────────────────────
    # Uses the same filtered dataset d and spatial weights W.
    dag_results = fit_dag_specs(
        y         = y,
        d         = d,
        W         = W,
        controls  = controls,
        sp_pretty = sp_pretty,
        h_lbl     = h_lbl,
    )
    plot_dag_sensitivity(
        dag_results = dag_results,
        health      = health,
        sp_pretty   = sp_pretty,
        outpath     = out_dir / f"{h_safe}_dag_sensitivity.png",
    )

    # Write DAG results to text summary
    with open(out_dir / f"{h_safe}_model_summary.txt", "a") as f:
        f.write(f"\n\n{'─'*65}\nDAG specification sensitivity:\n")
        for r in dag_results:
            f.write(f"\n  {r['spec_key']:20s}  R2_adj={r['R2_adj']:.4f}"
                    f"  beta_imperv_OLS={r['coef_imperv']:+.4f} (SE={r['se_imperv']:.4f})"
                    f"  beta_lst_OLS={r['coef_lst']:+.4f} (SE={r['se_lst']:.4f})")
        # Mediation indicator
        spec_keys_d = {r["spec_key"]: r for r in dag_results}
        if "Spec1_Full" in spec_keys_d and "Spec2_Urb" in spec_keys_d:
            b1 = spec_keys_d["Spec1_Full"]["coef_imperv"]
            b2 = spec_keys_d["Spec2_Urb"]["coef_imperv"]
            if np.isfinite(b1) and np.isfinite(b2) and abs(b2) > 1e-10:
                pct = (b1 - b2) / abs(b2) * 100
                f.write(f"\n  Beta_imperv Spec2->Spec1: {pct:+.1f}%"
                        f"  ({'possible mediation' if pct < -15 else 'direct effect likely'})")

    return rows, da_df.to_dict("records"), pr_export_rows, dag_results


# ╔══════════════════════════════════════════════════════════╗
# ║                       MAIN                              ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    all_rows = []
    all_da   = []
    all_pr   = []   # partial regression rows for combined plots
    all_dag  = []   # DAG specification rows for combined plots

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

        sp_rows, sp_da, sp_pr, sp_dag = [], [], [], []
        for health in HEALTH_VARS:
            print(f"  → {HEALTH_LABELS.get(health, health)}")
            h_out = sp_out / health
            ensure_dir(h_out)
            rows, da_rows, pr_rows, dag_rows = run_all_models(
                df, health, PREDICTORS, CONTROL_VARS, sp_pretty, h_out
            )
            sp_rows.extend(rows)
            sp_da.extend(da_rows)
            sp_pr.extend(pr_rows)
            sp_dag.extend(dag_rows)

        if not sp_rows:
            print("  [skip] No results produced."); continue

        pd.DataFrame(sp_rows).to_csv(sp_out / "model_comparison_all_metrics.csv", index=False)
        if sp_da:
            pd.DataFrame(sp_da).to_csv(sp_out / "dominance_all_metrics.csv", index=False)
        if sp_pr:
            pd.DataFrame(sp_pr).to_csv(sp_out / "partial_regression_all_metrics.csv", index=False)
        if sp_dag:
            pd.DataFrame(sp_dag).to_csv(sp_out / "dag_specs_all_metrics.csv", index=False)

        all_rows.extend(sp_rows)
        all_da.extend(sp_da)
        all_pr.extend(sp_pr)
        all_dag.extend(sp_dag)
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

    if all_pr:
        all_pr_df = pd.DataFrame(all_pr)
        all_pr_df.to_csv(OUT_ROOT / "ALL_species_partial_regression.csv", index=False)

        # Combined partial R² heatmap (averaged across species)
        plot_partial_r2_heatmap(all_pr_df, OUT_ROOT / "ALL_partial_R2_heatmap.png")

        # Partial R² vs dominance weight scatter (requires both datasets)
        if all_da:
            plot_partial_vs_dominance(
                all_pr_df,
                pd.DataFrame(all_da),
                OUT_ROOT / "ALL_partial_dominance_comparison.png"
            )

    if all_dag:
        all_dag_df = pd.DataFrame(all_dag)
        all_dag_df.to_csv(OUT_ROOT / "ALL_dag_specs.csv", index=False)
        plot_combined_dag_coef(all_dag_df, OUT_ROOT / "ALL_dag_coef_sensitivity.png")
        plot_combined_dag_r2(all_dag_df, OUT_ROOT / "ALL_dag_r2_comparison.png")

    if all_rows or all_da or all_pr or all_dag:
        print(f"\n[ok] All outputs saved to: {OUT_ROOT}")
    else:
        print("\n[warn] No results. Check paths, column names, and MIN_N.")


if __name__ == "__main__":
    main()