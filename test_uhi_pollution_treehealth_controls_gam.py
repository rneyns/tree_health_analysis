"""
Script: test_uhi_pollution_treehealth_controls_gam.py

Goal
- Investigate associations between tree health metrics and:
  (A) UHI proxy (LST variables)
  (B) Pollution proxy (NO2 variable)
- While controlling for tree height.
- Methods:
  1) Partial Spearman correlations (Pingouin): health ~ predictor | {height}
  2) GAMs (pyGAM): health ~ s(predictor)  + s(height)
  3) Added-variable style residual plots:
       resid(health | controls) vs resid(predictor | controls)
  4) GAM partial dependence plots for the focal predictor

Outputs
- CSV summary table with effect sizes + p-values (+ FDR)
- Plots folder:
    residual_plots/
    gam_partial_dependence/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import pingouin as pg

# pip install pygam
from pygam import LinearGAM, s

from scipy.stats import normaltest, shapiro, probplot



# ----------------------------
# USER SETTINGS
# ----------------------------

RAW_DATA_CSV = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/platanus x acerifolia/ndvi_metrics_with_impervious.csv"
OUT_DIR = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/platanus x acerifolia'

# Controls (always used)
CONTROL_VARS = ["height"]

# Focal predictors: UHI + pollution
UHI_VARS = ["lst_temp_r50_y", "lst_temp_r100_y"]
POLLUTION_VARS = ["poll_no2_anmean", "poll_bc_anmean", "poll_pm10_anmean", "poll_pm25_anmean"]

FOCAL_PREDICTORS = UHI_VARS + POLLUTION_VARS

# Health metrics: (suggested: deprioritize ndvi_peak/base if you want)
HEALTH_VARS = [
    "ndvi_peak",           # optional: often background-driven
    # "ndvi_base",           # optional: often background-driven
    "sos_doy",
    "los_days",
    "amplitude",
    "auc_above_base_full"
]

# GAM flexibility
N_SPLINES = 10
LAM_GRID = np.logspace(-3, 3, 15)  # smoothing search grid

# Plot settings
ALPHA = 0.30
FIG_DPI = 160

BIOLOGICAL_FILTERS = {
    "los_days": lambda x: (x >= 100) & (x <= 365),
    "sos_doy": lambda x: (x >= 1) & (x <= 200),
    "peak_doy": lambda x: (x >= 100) & (x <= 300),
    "eos_doy": lambda x: (x >= 200) & (x <= 365),
    "dormancy_doy": lambda x: (x >= 250) & (x <= 365),

    "amplitude": lambda x: (x > 0) & (x <= 1),
    "auc_above_base_full": lambda x: x > 0,
    "senescence_rate": lambda x: x > 0,
    "slope_sos_peak": lambda x: x > 0,

    "height": lambda x: x > 1,

    "poll_no2_anmean": lambda x: x > 0,
}

# Standardize (z-score) variables before partial corr / residual plots / GAM
STANDARDIZE = True

# Options:
# - "predictors_only": standardize pred + controls (health stays raw)
# - "all_model_vars": standardize health + pred + controls (makes added-variable slope ~ standardized beta)
STANDARDIZE_WHAT = "all_model_vars"

# Standardization style (keep simple)
STANDARDIZE_METHOD = "zscore"  # only zscore implemented below
STANDARDIZE_EPS = 1e-12


# ----------------------------
# HELPERS
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def residualize(y: pd.Series, X: pd.DataFrame) -> np.ndarray:
    """Return residuals of y ~ X (OLS with intercept)."""
    X_ = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y.values.astype(float), X_.values.astype(float), missing="drop").fit()
    return model.resid

def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def apply_biological_filters(df: pd.DataFrame, vars_involved: list[str]) -> pd.DataFrame:
    """
    Apply variable-specific biological validity filters
    only to the variables involved in the current analysis.
    """
    out = df.copy()
    for v in vars_involved:
        if v in BIOLOGICAL_FILTERS:
            mask = BIOLOGICAL_FILTERS[v](out[v])
            out = out[mask]
    return out


def plot_added_variable(df_sub: pd.DataFrame, health: str, pred: str, controls: list[str], outpath: str, standardized: bool = False) -> None:
    """
    Added-variable / partial regression style plot:
    residual(health | controls) vs residual(pred | controls)
    """
    Xc = df_sub[controls]
    ry = residualize(df_sub[health], Xc)
    rx = residualize(df_sub[pred], Xc)

    plt.figure()
    plt.scatter(rx, ry, alpha=ALPHA)

    m = sm.OLS(ry, sm.add_constant(rx, has_constant="add")).fit()
    xs = np.linspace(np.nanmin(rx), np.nanmax(rx), 200)
    ys = m.params[0] + m.params[1] * xs
    plt.plot(xs, ys)

    tag = " (standardized)" if standardized else ""
    plt.title(
        f"Added-variable plot{tag}: {health} ~ {pred} | {', '.join(controls)}\n"
        f"OLS slope={m.params[1]:.3g}, p={m.pvalues[1]:.3g}"
    )
    plt.xlabel(f"{pred} residuals | controls" + (" (z)" if standardized else ""))
    plt.ylabel(f"{health} residuals | controls" + (" (z)" if standardized else ""))
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


def fit_gam(df_sub: pd.DataFrame, health: str, pred: str, controls: list[str]):
    """
    Fit GAM:
      health ~ s(pred) + sum_i s(control_i)
    Uses gridsearch over lambda for smoothing.
    Returns fitted model + column order.
    """
    cols = [pred] + controls
    X = df_sub[cols].values.astype(float)
    y = df_sub[health].values.astype(float)

    # Build terms dynamically: s(0) for pred + s(1), s(2), ... for controls
    terms = s(0, n_splines=N_SPLINES)
    for j in range(1, len(cols)):
        terms = terms + s(j, n_splines=N_SPLINES)

    gam = LinearGAM(terms).gridsearch(X, y, lam=LAM_GRID, progress=False)
    return gam, cols


def plot_gam_partial_dependence(gam, cols: list[str], pred: str, outpath: str) -> None:
    """
    Partial dependence plot for focal predictor (term 0).
    Robust to pyGAM returning pdep/CI in different shapes across versions.
    Ensures x and y have matching lengths before plotting.
    """
    term_index = 0
    XX = gam.generate_X_grid(term=term_index)
    xgrid = np.asarray(XX[:, 0]).ravel()

    # --- Partial dependence (force to 1D) ---
    pdep = np.asarray(gam.partial_dependence(term=term_index, X=XX))
    # If pdep is (n,1) -> squeeze; if (n,2) -> take first col (most common culprit)
    if pdep.ndim == 2:
        if pdep.shape[1] == 1:
            pdep = pdep[:, 0]
        else:
            pdep = pdep[:, 0]
    pdep = np.asarray(pdep).ravel()

    # --- Confidence intervals (force to lower/upper 1D) ---
    ci = gam.partial_dependence(term=term_index, X=XX, width=0.95)

    lower = upper = None
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        lower = np.asarray(ci[0]).ravel()
        upper = np.asarray(ci[1]).ravel()
    else:
        ci_arr = np.asarray(ci)
        if ci_arr.ndim == 2 and ci_arr.shape[1] == 2:
            lower = ci_arr[:, 0].ravel()
            upper = ci_arr[:, 1].ravel()

    # --- Defensive alignment: make sure lengths match xgrid ---
    n = len(pdep)
    if len(xgrid) != n:
        # rebuild xgrid to match pdep length using the original range
        xgrid = np.linspace(np.nanmin(xgrid), np.nanmax(xgrid), n)

    if lower is not None and len(lower) != n:
        lower = np.interp(xgrid, np.linspace(xgrid.min(), xgrid.max(), len(lower)), lower)
    if upper is not None and len(upper) != n:
        upper = np.interp(xgrid, np.linspace(xgrid.min(), xgrid.max(), len(upper)), upper)

    # --- Plot ---
    plt.figure()
    plt.plot(xgrid, pdep)

    if lower is not None and upper is not None:
        plt.plot(xgrid, lower, linestyle="--")
        plt.plot(xgrid, upper, linestyle="--")

    plt.title(f"GAM partial dependence: {pred}\n(95% band if available)")
    plt.xlabel(pred)
    plt.ylabel("Partial effect on health")
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


def suggest_transform(x: np.ndarray) -> str:
    """
    Heuristic transform suggestion based on distribution shape.
    This is NOT automatic transformation; it just produces a note for the analyst.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return "n<20: inspect visually"

    s = pd.Series(x)
    skew = float(s.skew())
    # "right-skewed" vs "left-skewed" quick logic
    minv = float(np.min(x))

    # Very mild skew → no transform needed
    if abs(skew) < 0.5:
        return "none (approx symmetric)"

    # Right-skewed
    if skew >= 0.5:
        if minv > 0:
            if skew >= 2:
                return "log (or Box-Cox) (strong right skew)"
            return "log (or sqrt) (right skew)"
        if minv >= 0:
            return "log1p (right skew incl. zeros)"
        return "Yeo-Johnson (right skew incl. negatives)"

    # Left-skewed
    # often reflect/invert, or Yeo-Johnson
    if minv > 0:
        return "consider reflect + log/Box-Cox (left skew)"
    return "Yeo-Johnson (left skew incl. negatives)"


def plot_distributions_and_normality(
    df: pd.DataFrame,
    vars_to_check: list[str],
    out_dir: str,
    bins: int = 60,
    max_n_for_shapiro: int = 5000
) -> pd.DataFrame:
    """
    For each variable:
      - Histogram
      - Boxplot
      - Q–Q plot
      - Normality tests:
          * D’Agostino-Pearson K^2 (normaltest) for n>=8
          * Shapiro-Wilk (subsampled) for n>=3 (expensive at large n)
      - Distribution diagnostics: mean, sd, skew, kurtosis, suggested transform
    Returns a DataFrame summarizing diagnostics and tests.
    """
    ensure_dir(out_dir)

    rows = []
    for v in vars_to_check:
        x = pd.to_numeric(df[v], errors="coerce").dropna().values.astype(float)
        x = x[np.isfinite(x)]
        n = len(x)

        if n == 0:
            rows.append({"variable": v, "n": 0})
            continue

        # ---- Basic stats ----
        s = pd.Series(x)
        mean = float(s.mean())
        sd = float(s.std(ddof=1)) if n > 1 else np.nan
        skew = float(s.skew()) if n > 2 else np.nan
        kurt = float(s.kurtosis()) if n > 3 else np.nan
        tr_note = suggest_transform(x)

        # ---- Histogram ----
        plt.figure()
        plt.hist(x, bins=bins)
        plt.title(f"Histogram: {v} (n={n})")
        plt.xlabel(v)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, safe_filename(f"hist_{v}.png")), dpi=FIG_DPI)
        plt.close()

        # ---- Boxplot ----
        plt.figure()
        plt.boxplot(x, vert=True)
        plt.title(f"Boxplot: {v} (n={n})")
        plt.ylabel(v)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, safe_filename(f"box_{v}.png")), dpi=FIG_DPI)
        plt.close()

        # ---- Q–Q plot ----
        plt.figure()
        probplot(x, dist="norm", plot=plt)
        plt.title(f"Q–Q plot vs Normal: {v} (n={n})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, safe_filename(f"qq_{v}.png")), dpi=FIG_DPI)
        plt.close()

        # ---- Normality tests ----
        # D’Agostino-Pearson requires n>=8
        k2_stat = np.nan
        k2_p = np.nan
        if n >= 8:
            k2_stat, k2_p = normaltest(x)

        # Shapiro-Wilk (subsample for speed and because n is huge)
        sh_stat = np.nan
        sh_p = np.nan
        sh_n_used = int(min(n, max_n_for_shapiro))
        if n >= 3:
            if n > max_n_for_shapiro:
                rng = np.random.default_rng(42)
                xs = rng.choice(x, size=max_n_for_shapiro, replace=False)
            else:
                xs = x
            sh_stat, sh_p = shapiro(xs)

        rows.append({
            "variable": v,
            "n": n,
            "mean": mean,
            "sd": sd,
            "skew": skew,
            "kurtosis_excess": kurt,
            "suggested_transform": tr_note,
            "normaltest_k2": k2_stat,
            "normaltest_p": k2_p,
            "shapiro_w": sh_stat,
            "shapiro_p": sh_p,
            "shapiro_n_used": sh_n_used
        })

    return pd.DataFrame(rows)


def decide_transform(x: np.ndarray) -> str:
    """
    Decide a transform to TRY based on the distribution.
    Returns one of: 'none', 'log', 'log1p', 'yeojohnson', 'reflect_log', 'sqrt', 'reflect_sqrt'
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return "none"

    skew = float(pd.Series(x).skew())
    minv = float(np.min(x))

    if abs(skew) < 0.5:
        return "none"

    # right skew
    if skew >= 0.5:
        if minv > 0:
            return "log" if skew >= 1 else "sqrt"
        if minv >= 0:
            return "log1p"
        return "yeojohnson"

    # left skew
    if minv > 0:
        return "reflect_log" if abs(skew) >= 1 else "reflect_sqrt"
    return "yeojohnson"


def apply_transform(x: np.ndarray, how: str) -> tuple[np.ndarray, str]:
    """
    Apply a chosen transform. Returns (x_transformed, label).
    Safe for plotting/diagnostics (not for model fitting unless you choose to).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if how == "none":
        return x, "raw"

    if how == "log":
        # requires x>0
        xt = np.log(x)
        return xt, "log(x)"

    if how == "log1p":
        # requires x>=0
        xt = np.log1p(x)
        return xt, "log1p(x)"

    if how == "sqrt":
        # requires x>=0
        xt = np.sqrt(x)
        return xt, "sqrt(x)"

    if how == "reflect_log":
        # reflect around max then log
        m = np.max(x)
        xr = (m + 1e-9) - x
        xr = np.clip(xr, 1e-12, None)
        xt = np.log(xr)
        return xt, "log(max-x)"

    if how == "reflect_sqrt":
        m = np.max(x)
        xr = (m + 1e-9) - x
        xr = np.clip(xr, 0, None)
        xt = np.sqrt(xr)
        return xt, "sqrt(max-x)"

    if how == "yeojohnson":
        # Yeo-Johnson via sklearn (already in your env earlier)
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        xt = pt.fit_transform(x.reshape(-1, 1)).ravel()
        return xt, "yeo-johnson(x)"

    # fallback
    return x, "raw"


def plot_one_variable_set(
    x: np.ndarray,
    varname: str,
    out_dir: str,
    tag: str,
    bins: int,
    max_n_for_shapiro: int
) -> dict:
    """
    Create hist/box/qq plots + normality tests for a numeric array.
    Returns a dict of summary stats and test results.
    """
    ensure_dir(out_dir)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"variable": varname, "version": tag, "n": 0}

    s = pd.Series(x)
    mean = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else np.nan
    skew = float(s.skew()) if n > 2 else np.nan
    kurt = float(s.kurtosis()) if n > 3 else np.nan

    # Histogram
    plt.figure()
    plt.hist(x, bins=bins)
    plt.title(f"Histogram: {varname} [{tag}] (n={n})")
    plt.xlabel(varname)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, safe_filename(f"hist_{varname}__{tag}.png")), dpi=FIG_DPI)
    plt.close()

    # Boxplot
    plt.figure()
    plt.boxplot(x, vert=True)
    plt.title(f"Boxplot: {varname} [{tag}] (n={n})")
    plt.ylabel(varname)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, safe_filename(f"box_{varname}__{tag}.png")), dpi=FIG_DPI)
    plt.close()

    # Q–Q plot
    plt.figure()
    probplot(x, dist="norm", plot=plt)
    plt.title(f"Q–Q vs Normal: {varname} [{tag}] (n={n})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, safe_filename(f"qq_{varname}__{tag}.png")), dpi=FIG_DPI)
    plt.close()

    # Normality tests
    k2_stat = np.nan
    k2_p = np.nan
    if n >= 8:
        k2_stat, k2_p = normaltest(x)

    sh_stat = np.nan
    sh_p = np.nan
    sh_n_used = int(min(n, max_n_for_shapiro))
    if n >= 3:
        if n > max_n_for_shapiro:
            rng = np.random.default_rng(42)
            xs = rng.choice(x, size=max_n_for_shapiro, replace=False)
        else:
            xs = x
        sh_stat, sh_p = shapiro(xs)

    return {
        "variable": varname,
        "version": tag,
        "n": n,
        "mean": mean,
        "sd": sd,
        "skew": skew,
        "kurtosis_excess": kurt,
        "normaltest_k2": k2_stat,
        "normaltest_p": k2_p,
        "shapiro_w": sh_stat,
        "shapiro_p": sh_p,
        "shapiro_n_used": sh_n_used
    }


def plot_distributions_normality_with_transforms(
    df: pd.DataFrame,
    vars_to_check: list[str],
    out_dir: str,
    bins: int = 60,
    max_n_for_shapiro: int = 5000,
    do_transforms: bool = True
) -> pd.DataFrame:
    """
    For each variable, produce raw plots/tests and (optionally) transformed plots/tests.
    Returns a long-form table with rows for each variable x version (raw/transformed).
    """
    ensure_dir(out_dir)

    rows = []
    for v in vars_to_check:
        x_raw = pd.to_numeric(df[v], errors="coerce").dropna().values.astype(float)
        x_raw = x_raw[np.isfinite(x_raw)]
        if len(x_raw) == 0:
            rows.append({"variable": v, "version": "raw", "n": 0})
            continue

        # RAW
        rows.append(plot_one_variable_set(
            x=x_raw, varname=v, out_dir=out_dir, tag="raw",
            bins=bins, max_n_for_shapiro=max_n_for_shapiro
        ))

        # TRANSFORMED (suggested)
        transform_choice = decide_transform(x_raw)
        if do_transforms and transform_choice != "none":
            try:
                x_tr, label = apply_transform(x_raw, transform_choice)
                # Avoid degenerate transforms
                if np.isfinite(x_tr).sum() > 10 and np.nanstd(x_tr) > 0:
                    row_tr = plot_one_variable_set(
                        x=x_tr, varname=v, out_dir=out_dir, tag="transformed",
                        bins=bins, max_n_for_shapiro=max_n_for_shapiro
                    )
                    row_tr["transform_method"] = transform_choice
                    row_tr["transform_label"] = label
                    rows.append(row_tr)
            except Exception as e:
                # Keep pipeline robust: record failure but continue
                rows.append({
                    "variable": v,
                    "version": "transformed",
                    "n": len(x_raw),
                    "transform_method": transform_choice,
                    "transform_error": str(e)
                })

        # Also record the suggested transform even if not applied
        rows[-1].setdefault("transform_method", transform_choice)

    return pd.DataFrame(rows)

def zscore_series(s: pd.Series, eps: float = 1e-12) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd < eps:
        # avoid divide-by-zero; return centered zeros where possible
        return (s - mu) * 0.0
    return (s - mu) / sd

def maybe_standardize_subset(
    df_sub: pd.DataFrame,
    health: str,
    pred: str,
    controls: list[str],
    do: bool,
    what: str = "all_model_vars",
    method: str = "zscore",
    eps: float = 1e-12
) -> tuple[pd.DataFrame, str]:
    """
    Returns (df_out, tag) where tag is "" or "__z" for filenames/titles.
    Standardizes within df_sub (after NA drop + filtering), so plots/models match.
    """
    if not do:
        return df_sub, ""

    df_out = df_sub.copy()
    if method != "zscore":
        raise ValueError(f"Unsupported STANDARDIZE_METHOD: {method}")

    if what == "predictors_only":
        cols = [pred] + controls
    elif what == "all_model_vars":
        cols = [health, pred] + controls
    else:
        raise ValueError(f"Unsupported STANDARDIZE_WHAT: {what}")

    for c in cols:
        df_out[c] = zscore_series(df_out[c], eps=eps)

    return df_out, "__z"

# ----------------------------
# MAIN
# ----------------------------

def main():
    # -----------------------------------------
    # DISTRIBUTIONS + NORMALITY + Q–Q + TRANSFORMS
    # -----------------------------------------

    df = pd.read_csv(RAW_DATA_CSV)

    # Ensure numeric (avoid silent string columns)
    df = make_numeric(df, HEALTH_VARS + FOCAL_PREDICTORS + CONTROL_VARS)

    dist_dir = os.path.join(OUT_DIR, "variable_distributions")
    vars_to_check = list(dict.fromkeys(HEALTH_VARS + FOCAL_PREDICTORS + CONTROL_VARS))

    norm_res = plot_distributions_normality_with_transforms(
        df=df,
        vars_to_check=vars_to_check,
        out_dir=dist_dir,
        bins=60,
        max_n_for_shapiro=5000,
        do_transforms=True
    )

    norm_csv = os.path.join(OUT_DIR, "normality_tests_summary_long.csv")
    norm_res.to_csv(norm_csv, index=False)

    print(f"Saved plots to: {dist_dir}")
    print(f"Saved normality+transform summary to: {norm_csv}")


    ensure_dir(OUT_DIR)
    plots_resid_dir = os.path.join(OUT_DIR, "residual_plots")
    plots_gam_dir = os.path.join(OUT_DIR, "gam_partial_dependence")
    ensure_dir(plots_resid_dir)
    ensure_dir(plots_gam_dir)


    results = []

    for health in HEALTH_VARS:
        for pred in FOCAL_PREDICTORS:

            cols_needed = [health, pred] + CONTROL_VARS
            df_sub = df[cols_needed].dropna().copy()

            vars_involved = [health, pred] + CONTROL_VARS
            df_sub = apply_biological_filters(df_sub, vars_involved)

            if pred == "poll_no2_anmean":
                df_sub = df_sub[df_sub[pred] > 0].copy()

            # Guardrail: need enough rows
            n = len(df_sub)
            if n < 50:
                print(f"Skipping {health} ~ {pred}: only n={n}")
                continue

            # ----------------------------
            # OPTIONAL: STANDARDIZE
            # ----------------------------
            df_sub, ztag = maybe_standardize_subset(
                df_sub=df_sub,
                health=health,
                pred=pred,
                controls=CONTROL_VARS,
                do=STANDARDIZE,
                what=STANDARDIZE_WHAT,
                method=STANDARDIZE_METHOD,
                eps=STANDARDIZE_EPS
            )
            is_std = (ztag != "")


            # ----------------------------
            # 1) Partial Spearman (Pingouin)
            # ----------------------------
            pc = pg.partial_corr(
                data=df_sub,
                x=pred,
                y=health,
                covar=CONTROL_VARS,       # list of covariates
                method="spearman"
            )
            part_rho = float(pc["r"].iloc[0])
            part_p = float(pc["p-val"].iloc[0])

            # ----------------------------
            # 2) Added-variable residual plot
            # ----------------------------
            resid_plot_path = os.path.join(
                plots_resid_dir,
                safe_filename(f"resid_{health}__{pred}.png")
            )
            plot_added_variable(df_sub, health, pred, CONTROL_VARS, resid_plot_path)

            # ----------------------------
            # 3) GAM
            # ----------------------------
            gam, order = fit_gam(df_sub, health, pred, CONTROL_VARS)

            # pyGAM summary provides p-values per term, BUT see warnings in docs about p-values
            # when smoothing parameters are estimated (gridsearch). We'll still record them,
            # but you should treat them as approximate and lean on effect shape + CV/EDoF.
            stats = gam.statistics_
            # term 0 is focal pred smooth
            gam_p = np.nan
            if "p_values" in stats and len(stats["p_values"]) > 0:
                gam_p = float(stats["p_values"][0])

            # model fit metrics
            pseudo_r2 = np.nan
            if "pseudo_r2" in stats and "explained_deviance" in stats["pseudo_r2"]:
                pseudo_r2 = float(stats["pseudo_r2"]["explained_deviance"])

            edof = float(stats.get("edof", np.nan))

            # partial dependence plot for focal predictor
            gam_plot_path = os.path.join(
                plots_gam_dir,
                safe_filename(f"gam_pdep_{health}__{pred}.png")
            )
            plot_gam_partial_dependence(gam, order, pred, gam_plot_path)

            results.append({
                "health_metric": health,
                "predictor": pred,
                "predictor_type": "UHI" if pred in UHI_VARS else ("pollution" if pred in POLLUTION_VARS else "other"),
                "n": n,
                "partial_spearman_rho": part_rho,
                "partial_spearman_p": part_p,
                "gam_p_term0_approx": gam_p,
                "gam_pseudo_r2": pseudo_r2,
                "gam_edof": edof,
                "residual_plot": resid_plot_path,
                "gam_pdep_plot": gam_plot_path
            })

            print(f"Done: {health} ~ {pred} | controls  (n={n})")

    res = pd.DataFrame(results)

    # FDR adjustment across all partial correlation tests
    if len(res) > 0:
        res["partial_spearman_p_fdr"] = multipletests(res["partial_spearman_p"], method="fdr_bh")[1]

        # Optional: also FDR-adjust the approximate GAM term p-values (treat cautiously!)
        if res["gam_p_term0_approx"].notna().any():
            mask = res["gam_p_term0_approx"].notna()
            adj = np.full(len(res), np.nan)
            adj[mask] = multipletests(res.loc[mask, "gam_p_term0_approx"], method="fdr_bh")[1]
            res["gam_p_term0_fdr_approx"] = adj

    out_csv = os.path.join(OUT_DIR, "uhi_pollution_treehealth_summary.csv")
    res.to_csv(out_csv, index=False)

    print("\n✔ Done!")
    print(f"Saved summary: {out_csv}")
    print(f"Residual plots: {plots_resid_dir}")
    print(f"GAM partial dependence plots: {plots_gam_dir}")


if __name__ == "__main__":
    main()
