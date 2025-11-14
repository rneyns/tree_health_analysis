#!/usr/bin/env python3
"""
Compare NDVI curve fitting methods per tree:

1) Robust double logistic (your current approach)
2) Local Savitzky–Golay filtering
3) Harmonic (Fourier) model

Inputs (produced by your existing script)
-----------------------------------------
- <INPUT_DIR>/ndvi_samples_wide.csv
  - columns: tree_id, DOY1..DOYk
- <INPUT_DIR>/ndvi_wide_columns_mapping.csv
  - columns: col, date, doy, file

Outputs
-------
- <OUTPUT_DIR>/plots_compare/tree_<id>_ndvi_fit_compare.png
- <OUTPUT_DIR>/ndvi_fit_comparison_metrics.csv
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


# --------------------------- CONFIG ---------------------------
# Directory where ndvi_samples_wide.csv and ndvi_wide_columns_mapping.csv live
INPUT_DIR   = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope"

WIDE_CSV    = os.path.join(INPUT_DIR, "ndvi_samples_wide.csv")
MAPPING_CSV = os.path.join(INPUT_DIR, "ndvi_wide_columns_mapping.csv")

OUTPUT_DIR  = os.path.join(INPUT_DIR, "comparisons")
PLOT_DIR    = os.path.join(OUTPUT_DIR, "plots_compare")
METRICS_OUT = os.path.join(OUTPUT_DIR, "ndvi_fit_comparison_metrics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# General settings
MIN_OBS_TO_FIT   = 6
N_FIT_SAMPLES    = 2000
DAYS_FOR_PLOT    = 365.0

# Robust double logistic settings
ROBUST_LOSS      = "soft_l1"
ROBUST_F_SCALE   = 0.1
IRLS_REFINE      = True
IRLS_ITERS       = 5

# Savitzky–Golay settings (applied to a regular grid of length N_FIT_SAMPLES)
SAVGOL_WINDOW    = 5   # must be odd, <= N_FIT_SAMPLES
SAVGOL_POLYORDER = 3

# Harmonic model settings
N_HARMONICS      = 3    # number of sine/cosine pairs

# Penalized cubic smoothing spline settings
# This is a relative factor; actual s = SPLINE_LAMBDA * len(y_obs)
SPLINE_LAMBDA = 0.005   # tune this (e.g. 0.001–0.05)
# -------------------------------------------------------------


# ------------------------ Helpers ------------------------
def is_leap_year(year: int) -> bool:
    return (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0)


def date_to_doy_and_days(date_str):
    """
    Convert ISO date string (YYYY-MM-DD) to (doy, days_in_year).
    """
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    doy = int(d.strftime("%j"))
    days = 366 if is_leap_year(d.year) else 365
    return doy, days


def r2_score(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(m) < 2:
        return np.nan
    y_true = y_true[m]; y_pred = y_pred[m]
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(m) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m])**2)))


# ---------------- Double logistic model ----------------
def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    """
    t in [0,1]; parameters as in your original script.
    """
    s1 = 1.0 / (1.0 + np.exp(-mS * (t - S)))
    s2 = 1.0 / (1.0 + np.exp( mA * (t - A)))
    return wNDVI + (mNDVI - wNDVI) * (s1 + s2 - 1.0)


def robust_fit_curve(t, ndvi_observed, loss="soft_l1", f_scale=0.05, max_nfev=20000):
    """
    Fit double logistic model robustly as in your script.
    Returns parameter vector p or None.
    """
    t = np.asarray(t, float)
    y = np.asarray(ndvi_observed, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(y) < MIN_OBS_TO_FIT:
        return None

    w0 = float(np.nanpercentile(y, 10))
    m0 = float(np.nanpercentile(y, 90))
    S0 = float(np.clip(np.nanpercentile(t, 25), 0.05, 0.8))
    A0 = float(np.clip(np.nanpercentile(t, 75), S0 + 0.05, 0.95))
    p0 = np.array([w0, m0, S0, A0, 5.0, 5.0], dtype=float)
    lower = np.array([-1.0, -1.0, 0.00, 0.00, 0.01, 0.01], float)
    upper = np.array([ 1.50,  1.50, 1.00, 1.00, 50.0, 50.0], float)

    if p0[0] > p0[1]:
        p0[0], p0[1] = p0[1], p0[0]
    if p0[3] <= p0[2]:
        p0[3] = min(0.95, p0[2] + 0.1)

    def _res(p):
        return double_logistic_function(t, *p) - y

    try:
        res = least_squares(
            _res, p0,
            bounds=(lower, upper),
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev
        )
        p = res.x
        if p[3] <= p[2]:
            p[3] = min(0.99, p[2] + 0.05)
        return p
    except Exception:
        return None


def _tukey_biweight(resid, c=4.685):
    r = resid / (np.std(resid) + 1e-6)
    w = (1 - (r/c)**2)
    w[(r/c)**2 >= 1] = 0.0
    return np.clip(w, 0.0, 1.0)


def irls_refine(t, y, p_start, iters=3):
    """
    Iteratively reweighted least squares refinement for robustness (as in your script).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    p = np.array(p_start, float)
    lower = np.array([-1.0, -1.0, 0.00, 0.00, 0.01, 0.01], float)
    upper = np.array([ 1.50,  1.50, 1.00, 1.00, 50.0, 50.0], float)

    for _ in range(iters):
        resid = y - double_logistic_function(t, *p)
        w = _tukey_biweight(resid)
        if np.all(w == 0):
            break

        def _res(pp):
            return (double_logistic_function(t, *pp) - y) * np.sqrt(w)

        res = least_squares(
            _res, p,
            bounds=(lower, upper),
            loss="linear",
            max_nfev=8000
        )
        p = res.x
        if p[3] <= p[2]:
            p[3] = min(0.99, p[2] + 0.05)
    return p
# --------------------------------------------------------


# -------------- Savitzky–Golay model --------------------
def fit_savgol_on_grid(t, y, t_fit, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER):
    """
    - Interpolates irregular NDVI observations y(t) onto regular grid t_fit
    - Applies Savitzky–Golay filter on that grid.
    Returns y_fit_sg or None.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(y) < 3:
        return None

    # Sort by t
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # Interpolate to regular grid
    # Extrapolate by nearest endpoint
    y_interp = np.interp(t_fit, t, y, left=y[0], right=y[-1])

    # Ensure valid window length
    if window_length >= len(t_fit):
        window_length = len(t_fit) - 1 if len(t_fit) % 2 == 0 else len(t_fit)
    if window_length < polyorder + 2:
        window_length = polyorder + 2 + (polyorder + 2) % 2  # make odd
    if window_length < 5:
        # Not enough points for a meaningful filter – return interpolation
        return y_interp

    if window_length % 2 == 0:
        window_length += 1

    try:
        y_sg = savgol_filter(y_interp, window_length=window_length, polyorder=polyorder)
        return y_sg
    except Exception:
        return y_interp
# --------------------------------------------------------


# -------------- Harmonic (Fourier) model ----------------
def design_harmonic_matrix(t, n_harmonics):
    """
    Build design matrix for harmonic regression:
    y(t) = a0 + sum_k (a_k cos(2π k t) + b_k sin(2π k t))
    t in [0,1].
    """
    t = np.asarray(t, float)
    X_list = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        X_list.append(np.cos(2 * np.pi * k * t))
        X_list.append(np.sin(2 * np.pi * k * t))
    return np.vstack(X_list).T  # shape (n, 1 + 2*n_harmonics)


def fit_harmonic_model(t, y, n_harmonics=N_HARMONICS):
    """
    Fit harmonic (Fourier) regression model by linear least squares.
    Returns coefficients or None.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(y) < 2 * n_harmonics + 2:
        # Not enough points
        return None

    X = design_harmonic_matrix(t, n_harmonics)
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    except Exception:
        return None


def eval_harmonic_model(t_fit, beta, n_harmonics=N_HARMONICS):
    X_fit = design_harmonic_matrix(t_fit, n_harmonics)
    return X_fit @ beta
# --------------------------------------------------------

# -------------- Penalized cubic smoothing spline --------
def fit_spline_model(t, y, t_fit, lambda_rel=SPLINE_LAMBDA):
    """
    Penalized cubic smoothing spline:
      argmin sum (y_i - f(t_i))^2 + lambda * ∫ (f''(t))^2 dt

    Here we approximate lambda via the smoothing factor 's' used by
    UnivariateSpline: s ≈ lambda_rel * N, where N = number of obs.

    Parameters
    ----------
    t : array-like
        Time in [0,1].
    y : array-like
        NDVI values.
    t_fit : array-like
        Dense grid (0..1) on which to evaluate the spline.
    lambda_rel : float
        Relative smoothing factor; larger -> smoother spline.

    Returns
    -------
    spline : UnivariateSpline instance or None
    y_fit  : np.ndarray or None
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(y) < 5:
        return None, None

    # sort by time
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # smoothing factor: proportional to N
    s = lambda_rel * len(y)

    try:
        spline = UnivariateSpline(t, y, k=3, s=s)
        y_fit = spline(t_fit)
        return spline, y_fit
    except Exception:
        return None, None
# --------------------------------------------------------



# ---------------------------- MAIN ----------------------------
def main():
    # 1) Load NDVI wide + mapping
    df_wide = pd.read_csv(WIDE_CSV)
    df_map = pd.read_csv(MAPPING_CSV)

    # Ensure mapping sorted by col index (DOY1..DOYk)
    # Assumes mapping has 'col' like 'DOY1', 'DOY2', ...
    df_map = df_map.copy()
    df_map["col_index"] = df_map["col"].str.extract(r"DOY(\d+)", expand=False).astype(int)
    df_map = df_map.sort_values("col_index").reset_index(drop=True)

    # Build arrays of DOY and t_frac for each column
    # (t_frac per date = DOY / days_in_year)
    doys = []
    t_fracs = []
    for _, row in df_map.iterrows():
        doy, days_in_year = date_to_doy_and_days(row["date"])
        doys.append(doy)
        t_fracs.append(doy / float(days_in_year))
    doys = np.asarray(doys, float)
    t_fracs = np.asarray(t_fracs, float)

    doy_cols = df_map["col"].tolist()
    tree_id_col = "tree_id" if "tree_id" in df_wide.columns else df_wide.columns[0]

    # Prepare dense grid for fits (0..1)
    t_fit = np.linspace(0.0, 1.0, N_FIT_SAMPLES)
    doy_fit = t_fit * DAYS_FOR_PLOT

    metrics_rows = []

    # 2) Loop over trees
    for idx, row in df_wide.iterrows():
        tree_id = row[tree_id_col]

        y_obs = row[doy_cols].to_numpy(dtype=float)
        # mask finite obs
        m = np.isfinite(y_obs)
        if np.sum(m) < MIN_OBS_TO_FIT:
            print(f"[Skip] Tree {tree_id} – too few obs.")
            continue

        y_obs_valid = y_obs[m]
        t_valid = t_fracs[m]
        doy_valid = doys[m]

        # Double logistic
        p_dl = robust_fit_curve(t_valid, y_obs_valid, loss=ROBUST_LOSS, f_scale=ROBUST_F_SCALE)
        if p_dl is not None and IRLS_REFINE:
            p_dl = irls_refine(t_valid, y_obs_valid, p_dl, iters=IRLS_ITERS)

        if p_dl is not None:
            y_dl_fit = double_logistic_function(t_fit, *p_dl)
            y_dl_hat_obs = double_logistic_function(t_valid, *p_dl)
            rmse_dl = rmse(y_obs_valid, y_dl_hat_obs)
            r2_dl = r2_score(y_obs_valid, y_dl_hat_obs)
        else:
            y_dl_fit = None
            rmse_dl = np.nan
            r2_dl = np.nan

        # Savitzky–Golay smoothing
        y_sg_fit = fit_savgol_on_grid(t_valid, y_obs_valid, t_fit)
        if y_sg_fit is not None:
            # Interpolate sg curve back to observation times for metrics
            y_sg_hat_obs = np.interp(t_valid, t_fit, y_sg_fit)
            rmse_sg = rmse(y_obs_valid, y_sg_hat_obs)
            r2_sg = r2_score(y_obs_valid, y_sg_hat_obs)
        else:
            rmse_sg = np.nan
            r2_sg = np.nan

        # Harmonic model
        beta = fit_harmonic_model(t_valid, y_obs_valid, n_harmonics=N_HARMONICS)
        if beta is not None:
            y_harm_fit = eval_harmonic_model(t_fit, beta, n_harmonics=N_HARMONICS)
            y_harm_hat_obs = eval_harmonic_model(t_valid, beta, n_harmonics=N_HARMONICS)
            rmse_harm = rmse(y_obs_valid, y_harm_hat_obs)
            r2_harm = r2_score(y_obs_valid, y_harm_hat_obs)
        else:
            y_harm_fit = None
            rmse_harm = np.nan
            r2_harm = np.nan

        # Penalized cubic smoothing spline
        spline, y_spline_fit = fit_spline_model(t_valid, y_obs_valid, t_fit, lambda_rel=SPLINE_LAMBDA)
        if spline is not None and y_spline_fit is not None:
            y_spline_hat_obs = spline(t_valid)
            rmse_spline = rmse(y_obs_valid, y_spline_hat_obs)
            r2_spline = r2_score(y_obs_valid, y_spline_hat_obs)
        else:
            rmse_spline = np.nan
            r2_spline = np.nan


        # Store metrics rows
        metrics_rows.append({
            "tree_id": tree_id,
            "n_obs": int(np.sum(m)),
            "rmse_double_logistic": rmse_dl,
            "r2_double_logistic": r2_dl,
            "rmse_savgol": rmse_sg,
            "r2_savgol": r2_sg,
            "rmse_harmonic": rmse_harm,
            "r2_harmonic": r2_harm,
            "rmse_spline": rmse_spline,
            "r2_spline": r2_spline,
        })


        # 3) Plot comparison
        plt.figure(figsize=(8.5, 5.0))
        # Observed points
        plt.scatter(doy_valid, y_obs_valid, s=25, alpha=0.9, label="Observed NDVI")

        if y_dl_fit is not None:
            plt.plot(doy_fit, y_dl_fit, linewidth=2, label=f"Double logistic (RMSE={rmse_dl:.3f})")
        if y_sg_fit is not None:
            plt.plot(doy_fit, y_sg_fit, linewidth=1.8, linestyle="--", label=f"Savitzky–Golay (RMSE={rmse_sg:.3f})")
        if beta is not None:
            plt.plot(doy_fit, y_harm_fit, linewidth=1.8, linestyle="-.", label=f"Harmonic ({N_HARMONICS} harm., RMSE={rmse_harm:.3f})")
        if y_spline_fit is not None:
            plt.plot(
                doy_fit,
                y_spline_fit,
                linewidth=1.8,
                linestyle=":",
                label=f"Smoothing spline (RMSE={rmse_spline:.3f})"
            )

        plt.xlabel("Day of Year")
        plt.ylabel("NDVI")
        plt.title(f"Tree {tree_id} — NDVI curve comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()

        out_png = os.path.join(PLOT_DIR, f"tree_{tree_id}_ndvi_fit_compare.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[Saved] {out_png}")

    # 4) Save metrics CSV
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(METRICS_OUT, index=False)
    print(f"[OK] Saved comparison metrics: {METRICS_OUT}")
    print("[Done]")


if __name__ == "__main__":
    main()
