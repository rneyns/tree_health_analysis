#!/usr/bin/env python3
"""
Parameter sweep for double-logistic NDVI fitting.

- Loads ndvi_samples_wide.csv + ndvi_wide_columns_mapping.csv
- Reconstructs per-tree NDVI time series (t, DOY, NDVI)
- Randomly selects N trees
- Fits multiple parameter configurations (loss / f_scale / IRLS)
- Plots observed NDVI + all fits for each tree

Output:
  <OUTPUT_PLOT_DIR>/tree_<tree_id>_param_sweep.png
"""

import os
import re
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ============================ CONFIG ============================

# Paths to your existing outputs
BASE_DIR = Path("/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope")
WIDE_CSV_PATH    = BASE_DIR / "ndvi_samples_wide.csv"
MAPPING_CSV_PATH = BASE_DIR / "ndvi_wide_columns_mapping.csv"  # ndvi_wide_columns_mapping.csv
OUTPUT_PLOT_DIR  = BASE_DIR / "param_sweep_plots"

OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Column name for tree ID in wide CSV
TREE_ID_COL = "tree_id"

# Minimum observations to attempt a fit
MIN_OBS_TO_FIT = 6

# Number of trees to sample for visual inspection
N_TREES_SAMPLE = 30

RANDOM_SEED = 42  # for reproducibility

# Parameter configurations to compare
# Each entry defines how robust_fit_curve() + (optional) IRLS should behave
PARAM_CONFIGS = [
    {
        "name": "linear_no_irls",
        "loss": "linear",
        "f_scale": 1.0,
        "irls_refine": False,
        "irls_iters": 0,
    },
    {
        "name": "softl1_f0.10_no_irls",
        "loss": "soft_l1",
        "f_scale": 0.10,
        "irls_refine": False,
        "irls_iters": 0,
    },
    {
        "name": "softl1_f0.10_irls2",
        "loss": "soft_l1",
        "f_scale": 0.10,
        "irls_refine": True,
        "irls_iters": 2,
    },
    {
        "name": "softl1_f0.20_irls3",
        "loss": "soft_l1",
        "f_scale": 0.20,
        "irls_refine": True,
        "irls_iters": 3,
    },
]

# Colors/linestyles for plotting each config (cycled if fewer specified)
CONFIG_STYLES = [
    {"color": "tab:blue",   "linestyle": "-"},
    {"color": "tab:orange", "linestyle": "--"},
    {"color": "tab:green",  "linestyle": "-."},
    {"color": "tab:red",    "linestyle": ":"},
]

# Number of samples for smooth curve
N_FIT_SAMPLES = 1000
DAYS_FOR_PLOT = 365.0

# ================================================================


# ---------------- Double logistic + robust fitting --------------

def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    """Standard double logistic NDVI model."""
    s1 = 1.0 / (1.0 + np.exp(-mS * (t - S)))
    s2 = 1.0 / (1.0 + np.exp( mA * (t - A)))
    return wNDVI + (mNDVI - wNDVI) * (s1 + s2 - 1.0)


def robust_fit_curve(t, ndvi_observed, loss="soft_l1", f_scale=0.05, max_nfev=20000):
    """
    Fit double logistic using scipy.least_squares with robust loss.

    Parameters
    ----------
    t : array-like, float
        Time in [0,1].
    ndvi_observed : array-like, float
        Observed NDVI values.
    loss : str
        'linear', 'soft_l1', 'huber', 'cauchy', etc. (SciPy)
    f_scale : float
        Scale parameter: residuals ~< f_scale considered "inliers".

    Returns
    -------
    p : array or None
        Fitted parameters [wNDVI, mNDVI, S, A, mS, mA], or None if fit fails.
    """
    t = np.asarray(t, float)
    y = np.asarray(ndvi_observed, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(y) < MIN_OBS_TO_FIT:
        return None

    # Initial guess based on percentiles
    w0 = float(np.nanpercentile(y, 10))  # base NDVI
    m0 = float(np.nanpercentile(y, 90))  # max NDVI

    S0 = float(np.clip(np.nanpercentile(t, 25), 0.05, 0.8))
    A0 = float(np.clip(np.nanpercentile(t, 75), S0 + 0.05, 0.95))

    p0 = np.array([w0, m0, S0, A0, 5.0, 5.0], dtype=float)

    lower = np.array([-1.0, -1.0, 0.00, 0.00, 0.01, 0.01], float)
    upper = np.array([ 1.50,  1.50, 1.00, 1.00, 50.0, 50.0], float)

    # Ensure w <= m
    if p0[0] > p0[1]:
        p0[0], p0[1] = p0[1], p0[0]
    # Ensure A>S
    if p0[3] <= p0[2]:
        p0[3] = min(0.95, p0[2] + 0.1)

    def _res(p):
        return double_logistic_function(t, *p) - y

    try:
        res = least_squares(
            _res,
            p0,
            bounds=(lower, upper),
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        p = res.x
        # enforce A>S again
        if p[3] <= p[2]:
            p[3] = min(0.99, p[2] + 0.05)
        return p
    except Exception:
        return None


def _tukey_biweight(resid, c=4.685):
    """
    Tukey's biweight weights for IRLS.
    resid: residuals y - y_hat
    c: tuning constant ~4.685 for 95% efficiency (classic choice)
    """
    r = resid / (np.std(resid) + 1e-6)
    w = (1 - (r / c) ** 2)
    w[(r / c) ** 2 >= 1] = 0.0
    return np.clip(w, 0.0, 1.0)


def irls_refine(t, y, p_start, iters=3):
    """
    Iteratively re-weighted least squares refinement using Tukey biweight.
    Uses 'linear' loss inside least_squares but multiplies residuals by sqrt(w).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    p = np.array(p_start, float)

    lower = np.array([-1.0, -1.0, 0.00, 0.00, 0.01, 0.01], float)
    upper = np.array([ 1.50,  1.50, 1.00, 1.00, 50.0, 50.0], float)

    for _ in range(iters):
        y_hat = double_logistic_function(t, *p)
        resid = y - y_hat
        w = _tukey_biweight(resid)
        if np.all(w == 0):
            break

        def _res(pp):
            return (double_logistic_function(t, *pp) - y) * np.sqrt(w)

        try:
            res = least_squares(
                _res,
                p,
                bounds=(lower, upper),
                loss="linear",
                max_nfev=8000,
            )
            p = res.x
            if p[3] <= p[2]:
                p[3] = min(0.99, p[2] + 0.05)
        except Exception:
            break
    return p

# ---------------------------------------------------------------


def days_in_year(year: int) -> int:
    """Return number of days in year (365 or 366)."""
    if (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
        return 366
    return 365


def load_ndvi_wide_and_mapping():
    """Load wide NDVI + DOY mapping and build a per-column time mapping."""
    df_wide = pd.read_csv(WIDE_CSV_PATH)
    mapping = pd.read_csv(MAPPING_CSV_PATH)

    # Expect mapping columns: col, date, doy, file
    # Build dict: col_name -> (t_frac, doy)
    col_to_t_doy = {}
    for _, row in mapping.iterrows():
        col_name = row["col"]
        date_str = row["date"]
        doy = int(row["doy"])
        try:
            dt = datetime.fromisoformat(date_str).date()
            n_days = days_in_year(dt.year)
            t_frac = doy / float(n_days)
        except Exception:
            # fallback: assume 365
            t_frac = doy / 365.0
        col_to_t_doy[col_name] = (t_frac, doy)

    # Identify DOY* columns
    doy_cols = [c for c in df_wide.columns if c.startswith("DOY")]
    return df_wide, doy_cols, col_to_t_doy


def build_per_tree_time_series(df_wide, doy_cols, col_to_t_doy):
    """
    From wide NDVI table, build per-tree lists:
      - t_list[tree_idx]     : list of t in [0,1]
      - doy_list[tree_idx]   : list of DOY
      - ndvi_list[tree_idx]  : list of NDVI values
    """
    n_trees = df_wide.shape[0]
    per_tree_t = [[] for _ in range(n_trees)]
    per_tree_doy = [[] for _ in range(n_trees)]
    per_tree_ndvi = [[] for _ in range(n_trees)]

    for i in range(n_trees):
        row = df_wide.iloc[i]
        for col in doy_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if col not in col_to_t_doy:
                continue
            t_frac, doy = col_to_t_doy[col]
            per_tree_t[i].append(float(t_frac))
            per_tree_doy[i].append(float(doy))
            per_tree_ndvi[i].append(float(val))

    return per_tree_t, per_tree_doy, per_tree_ndvi


def select_random_trees(per_tree_ndvi, n_select=30, min_obs=MIN_OBS_TO_FIT, seed=42):
    """Randomly select tree indices with at least min_obs observations."""
    random.seed(seed)
    candidates = [i for i, ys in enumerate(per_tree_ndvi) if np.sum(np.isfinite(ys)) >= min_obs]
    if len(candidates) <= n_select:
        return candidates
    return random.sample(candidates, n_select)


def main():
    print(f"[INFO] Loading NDVI wide from {WIDE_CSV_PATH}")
    print(f"[INFO] Loading DOY mapping from {MAPPING_CSV_PATH}")
    df_wide, doy_cols, col_to_t_doy = load_ndvi_wide_and_mapping()

    if TREE_ID_COL in df_wide.columns:
        tree_ids = df_wide[TREE_ID_COL].tolist()
    else:
        tree_ids = list(range(df_wide.shape[0]))

    per_tree_t, per_tree_doy, per_tree_ndvi = build_per_tree_time_series(
        df_wide, doy_cols, col_to_t_doy
    )
    n_trees = len(per_tree_t)
    print(f"[INFO] Built time series for {n_trees} trees.")

    # Pick random subset
    sample_indices = select_random_trees(
        per_tree_ndvi,
        n_select=N_TREES_SAMPLE,
        min_obs=MIN_OBS_TO_FIT,
        seed=RANDOM_SEED,
    )
    print(f"[INFO] Selected {len(sample_indices)} trees for parameter sweep plots.")

    # Precompute t_fit for smooth curves
    t_fit = np.linspace(0.0, 1.0, N_FIT_SAMPLES)
    x_fit_days = t_fit * DAYS_FOR_PLOT

    # Loop over sample trees
    for idx, tid in enumerate(sample_indices):
        t = np.asarray(per_tree_t[tid], float)
        y = np.asarray(per_tree_ndvi[tid], float)
        doy = np.asarray(per_tree_doy[tid], float)

        m = np.isfinite(t) & np.isfinite(y) & np.isfinite(doy)
        t, y, doy = t[m], y[m], doy[m]

        if len(y) < MIN_OBS_TO_FIT:
            print(f"[WARN] Tree {tid} has fewer than {MIN_OBS_TO_FIT} obs after filtering; skipping.")
            continue

        tree_label = tree_ids[tid]
        print(f"[TREE {idx+1}/{len(sample_indices)}] index={tid}, tree_id={tree_label}, n_obs={len(y)}")

        plt.figure(figsize=(8, 5))

        # Plot observed NDVI
        plt.scatter(doy, y, s=30, color="k", alpha=0.8, label="Observed NDVI")

        # Try each parameter configuration
        for cfg_idx, cfg in enumerate(PARAM_CONFIGS):
            style = CONFIG_STYLES[cfg_idx % len(CONFIG_STYLES)]
            name = cfg["name"]
            loss = cfg["loss"]
            f_scale = cfg["f_scale"]
            irls_refine_flag = cfg["irls_refine"]
            irls_iters = cfg["irls_iters"]

            p = robust_fit_curve(t, y, loss=loss, f_scale=f_scale)
            if p is None:
                print(f"   [CFG {name}] fit failed (initial).")
                continue

            if irls_refine_flag and irls_iters > 0:
                p = irls_refine(t, y, p, iters=irls_iters)

            # Evaluate smooth curve
            y_fit = double_logistic_function(t_fit, *p)
            label = f"{name} (loss={loss}, f={f_scale}, IRLS={irls_iters if irls_refine_flag else 0})"
            plt.plot(
                x_fit_days,
                y_fit,
                label=label,
                linewidth=1.8,
                **style,
            )

        plt.xlabel("Day of Year")
        plt.ylabel("NDVI")
        plt.title(f"Tree {tree_label} (index {tid}) – parameter sweep")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)

        out_png = OUTPUT_PLOT_DIR / f"tree_{tree_label}_param_sweep.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"   [SAVED] {out_png}")

    print("[DONE] Parameter sweep plots written to:", OUTPUT_PLOT_DIR)


if __name__ == "__main__":
    main()
