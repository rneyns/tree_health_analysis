"""
Script: test_health_environment_relationships.py

Purpose:
- Test whether any tree health metric is meaningfully correlated with any environmental variable.
- Tests include Pearson, Spearman, Kendall, Partial Correlation, and Distance Correlation.
- Outputs a table summarizing effect sizes + significance with FDR correction.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import pingouin as pg  # pip install pingouin (for partial correlations)
from dcor import distance_correlation  # pip install dcor

# ----------------------------
# USER INPUT
# ----------------------------

df = pd.read_csv('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/ndvi_metrics_with_impervious.csv')

# Define your health metrics
health_vars = [
    "ndvi_peak",
    "ndvi_base",
    "greenup_doy",
    "sos_doy",
    "peak_doy",
    "sen_onset_doy",
    "eos_doy",
    "dormancy_doy",
    "los_days",
    "amplitude",
    "slope_sos_peak",
    "senescence_rate",
    "auc_above_base_full"
]

# Define your environmental predictors
env_vars = [
    "imperv_10m",
    "imperv_20m",
    "poll_no2_anmean",
    "lst_temp_r100",
    "lst_temp_r50",
    "height",
]

# Variable to CONTROL FOR in partial correlations
control_var = "imperv_10m"

# ----------------------------
# RESULTS TABLE
# ----------------------------

results = []

for h in health_vars:
    for e in env_vars:

        if h == e:
            continue

        # Drop NA for the three vars we need
        df_sub = df[[h, e, control_var]].dropna()

        # Convert to numpy arrays
        x = np.asarray(df_sub[h])
        y = np.asarray(df_sub[e])

        # --- Guardrails: make sure both are 1D numeric vectors ---
        # Skip if y (or x) is multi-dimensional, e.g. shape (n, 2)
        if x.ndim != 1 or y.ndim != 1:
            print(f"Skipping pair {h} – {e}: x.shape={x.shape}, y.shape={y.shape} (need 1D)")
            continue

        # Skip if lengths don't match for some reason
        if x.shape[0] != y.shape[0]:
            print(f"Skipping pair {h} – {e}: length mismatch {x.shape[0]} vs {y.shape[0]}")
            continue

        # Try casting to float; skip if it fails (non-numeric)
        try:
            x = x.astype(float)
            y = y.astype(float)
        except ValueError:
            print(f"Skipping pair {h} – {e}: could not convert to float")
            continue

        # -----------------------------
        # Compute correlations
        # -----------------------------
        pear_r, pear_p = pearsonr(x, y)
        spear_rho, spear_p = spearmanr(x, y)
        kend_tau, kend_p = kendalltau(x, y)

        # Partial Spearman: health ~ env | control_var
        pcorr = pg.partial_corr(
            data=df_sub,
            x=h,
            y=e,
            covar=control_var,
            method="spearman"
        )
        part_rho = pcorr["r"].values[0]
        part_p = pcorr["p-val"].values[0]

        # Distance correlation
        dcor_val = distance_correlation(x, y)

        results.append({
            "health_metric": h,
            "env_metric": e,
            "pearson_r": pear_r, "pearson_p": pear_p,
            "spearman_rho": spear_rho, "spearman_p": spear_p,
            "kendall_tau": kend_tau, "kendall_p": kend_p,
            "partial_spearman_rho": part_rho, "partial_p": part_p,
            "distance_corr": dcor_val
        })


# ----------------------------
# CREATE RESULTS DATAFRAME
# ----------------------------

res = pd.DataFrame(results)

# FDR correction across all tests
for col in ["pearson_p", "spearman_p", "kendall_p", "partial_p"]:
    res[col+"_adj"] = multipletests(res[col], method='fdr_bh')[1]

res.to_csv('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/health_environment_stats_summary.csv', index=False)

print("✔ Done! Results saved.")

#-------------------------------------------
# Plotting the most interesting correlations
#--------------------------------------------

df_raw = df

import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(
    data=df_raw,
    x="imperv_20m",
    y="ndvi_peak",
    lowess=True,
    scatter_kws={"alpha": 0.3}
)

plt.title("NDVI peak vs impervious surface (20 m)")
plt.xlabel("Impervious surface fraction (20 m)")
plt.ylabel("NDVI peak")
plt.show()

sns.regplot(
    data=df_raw,
    x="imperv_10m",
    y="ndvi_peak",
    lowess=True,
    scatter_kws={"alpha": 0.3},
    label="10 m"
)

sns.regplot(
    data=df_raw,
    x="imperv_20m",
    y="ndvi_peak",
    lowess=True,
    scatter_kws={"alpha": 0.3},
    label="20 m",
    color="red"
)

plt.legend()
plt.title("NDVI peak vs imperviousness at two spatial scales")
plt.show()

sns.regplot(
    data=df_raw,
    x="height",
    y="ndvi_peak",
    lowess=True,
    scatter_kws={"alpha": 0.3}
)

plt.title("NDVI peak vs tree height")
plt.xlabel("Tree height (m)")
plt.ylabel("NDVI peak")
plt.show()


from pingouin import partial_corr

# residual plot logic
import statsmodels.api as sm

X = sm.add_constant(df_raw["imperv_20m"])
res_ndvi = sm.OLS(df_raw["ndvi_peak"], X).fit().resid
res_height = sm.OLS(df_raw["height"], X).fit().resid

plt.scatter(res_height, res_ndvi, alpha=0.3)
plt.xlabel("Height (residuals | imperv_20m)")
plt.ylabel("NDVI peak (residuals | imperv_20m)")
plt.title("Partial relationship: NDVI peak ~ height | imperviousness")
plt.show()

sns.scatterplot(
    data=df_raw,
    x="poll_no2_anmean",
    y="ndvi_peak",
    alpha=0.3
)

sns.regplot(
    data=df_raw,
    x="poll_no2_anmean",
    y="ndvi_peak",
    lowess=True,
    scatter=False,
    color="red"
)

plt.title("NDVI peak vs NO₂ (possible nonlinear response)")
plt.show()


# -------------------------------------------
# Composite figure: Pollution vs Heat (GAM + residual plots)
# -------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# If you don't already have these installed:
# pip install pygam
from pygam import LinearGAM, s

# ---- Settings (edit if needed) ----
HEALTH = "auc_above_base_full"
PRED_NO2 = "poll_no2_anmean"
PRED_LST = "lst_temp_r100"      # choose r50 or r100
CONTROLS = ["imperv_10m", "height"]

# NO2 cleaning (applied only to NO2)
NO2_MIN_VALID = 0.0   # set to >0 if 0 is invalid in your dataset
TRIM_NO2_TAILS = True
NO2_LOW_Q = 0.01
NO2_HIGH_Q = 0.99

# GAM smoothness
N_SPLINES = 10
LAM_GRID = np.logspace(-3, 3, 15)

ALPHA = 0.25

def _make_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals from OLS y ~ X (with intercept)."""
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc, missing="drop").fit().resid

def _prep_subset(df, y_col, x_col, controls, apply_no2_filter=False):
    cols = [y_col, x_col] + controls
    d = df[cols].dropna().copy()
    d = _make_numeric(d, cols).dropna()

    if apply_no2_filter:
        d = d[d[x_col] > NO2_MIN_VALID].copy()
        if TRIM_NO2_TAILS and len(d) > 0:
            lo = d[x_col].quantile(NO2_LOW_Q)
            hi = d[x_col].quantile(NO2_HIGH_Q)
            d = d[(d[x_col] >= lo) & (d[x_col] <= hi)].copy()

    return d

def _fit_gam_and_grid(d, y_col, x_col, controls):
    """
    Fit GAM: y ~ s(x) + s(control1) + s(control2)
    Returns: gam, xgrid, pdep, (lower, upper)
    """
    cols = [x_col] + controls
    X = d[cols].values.astype(float)
    y = d[y_col].values.astype(float)

    gam = LinearGAM(
        s(0, n_splines=N_SPLINES) + s(1, n_splines=N_SPLINES) + s(2, n_splines=N_SPLINES)
    ).gridsearch(X, y, lam=LAM_GRID, progress=False)

    # grid for term 0 (focal predictor)
    XX = gam.generate_X_grid(term=0)
    xgrid = XX[:, 0]

    pdep = gam.partial_dependence(term=0, X=XX)
    pdep = np.asarray(pdep)
    if pdep.ndim == 2:
        pdep = pdep[:, 0]
    pdep = pdep.ravel()

    # confidence bands (pyGAM versions differ)
    ci = gam.partial_dependence(term=0, X=XX, width=0.95)
    lower = upper = None

    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        lower = np.asarray(ci[0]).ravel()
        upper = np.asarray(ci[1]).ravel()
    else:
        ci_arr = np.asarray(ci)
        if ci_arr.ndim == 2 and ci_arr.shape[1] == 2:
            lower = ci_arr[:, 0].ravel()
            upper = ci_arr[:, 1].ravel()

    # Defensive alignment (in case a version returns odd shapes)
    if len(xgrid) != len(pdep):
        xgrid = np.linspace(np.nanmin(xgrid), np.nanmax(xgrid), len(pdep))

    return gam, xgrid, pdep, lower, upper

def _added_variable_data(d, y_col, x_col, controls):
    """
    Compute residual(y|controls) and residual(x|controls)
    """
    yc = d[y_col].values.astype(float)
    xc = d[x_col].values.astype(float)
    Z = d[controls].values.astype(float)

    ry = _residualize(yc, Z)
    rx = _residualize(xc, Z)
    return rx, ry

# ---- Prepare data ----
d_no2 = _prep_subset(df, HEALTH, PRED_NO2, CONTROLS, apply_no2_filter=True)
d_lst = _prep_subset(df, HEALTH, PRED_LST, CONTROLS, apply_no2_filter=False)

# ---- Fit GAMs ----
gam_no2, x_no2, p_no2, lo_no2, hi_no2 = _fit_gam_and_grid(d_no2, HEALTH, PRED_NO2, CONTROLS)
gam_lst, x_lst, p_lst, lo_lst, hi_lst = _fit_gam_and_grid(d_lst, HEALTH, PRED_LST, CONTROLS)

# ---- Added-variable residual data ----
rx_no2, ry_no2 = _added_variable_data(d_no2, HEALTH, PRED_NO2, CONTROLS)
rx_lst, ry_lst = _added_variable_data(d_lst, HEALTH, PRED_LST, CONTROLS)

# ---- Build composite figure ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (A) NO2 GAM partial dependence
ax = axes[0, 0]
ax.plot(x_no2, p_no2)
if lo_no2 is not None and hi_no2 is not None and len(lo_no2) == len(x_no2) and len(hi_no2) == len(x_no2):
    ax.plot(x_no2, lo_no2, linestyle="--")
    ax.plot(x_no2, hi_no2, linestyle="--")
ax.set_title(f"(A) GAM partial effect: {PRED_NO2} → {HEALTH}\ncontrols: {', '.join(CONTROLS)}")
ax.set_xlabel(PRED_NO2)
ax.set_ylabel("Partial effect on health")

# (B) NO2 added-variable plot
ax = axes[0, 1]
ax.scatter(rx_no2, ry_no2, alpha=ALPHA)
m = sm.OLS(ry_no2, sm.add_constant(rx_no2, has_constant="add")).fit()
xs = np.linspace(np.nanmin(rx_no2), np.nanmax(rx_no2), 200)
ax.plot(xs, m.params[0] + m.params[1] * xs)
ax.set_title(f"(B) Added-variable: {PRED_NO2} residual vs {HEALTH} residual\n"
             f"slope={m.params[1]:.3g}, p={m.pvalues[1]:.3g}")
ax.set_xlabel(f"{PRED_NO2} residuals | controls")
ax.set_ylabel(f"{HEALTH} residuals | controls")

# (C) LST GAM partial dependence
ax = axes[1, 0]
ax.plot(x_lst, p_lst)
if lo_lst is not None and hi_lst is not None and len(lo_lst) == len(x_lst) and len(hi_lst) == len(x_lst):
    ax.plot(x_lst, lo_lst, linestyle="--")
    ax.plot(x_lst, hi_lst, linestyle="--")
ax.set_title(f"(C) GAM partial effect: {PRED_LST} → {HEALTH}\ncontrols: {', '.join(CONTROLS)}")
ax.set_xlabel(PRED_LST)
ax.set_ylabel("Partial effect on health")

# (D) LST added-variable plot
ax = axes[1, 1]
ax.scatter(rx_lst, ry_lst, alpha=ALPHA)
m2 = sm.OLS(ry_lst, sm.add_constant(rx_lst, has_constant="add")).fit()
xs2 = np.linspace(np.nanmin(rx_lst), np.nanmax(rx_lst), 200)
ax.plot(xs2, m2.params[0] + m2.params[1] * xs2)
ax.set_title(f"(D) Added-variable: {PRED_LST} residual vs {HEALTH} residual\n"
             f"slope={m2.params[1]:.3g}, p={m2.pvalues[1]:.3g}")
ax.set_xlabel(f"{PRED_LST} residuals | controls")
ax.set_ylabel(f"{HEALTH} residuals | controls")

plt.tight_layout()

# Save (recommended for paper), or just show
# fig.savefig("composite_NO2_vs_LST_auc_controls.png", dpi=300, bbox_inches="tight")
plt.show()

