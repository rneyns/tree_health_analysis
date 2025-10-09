#!/usr/bin/env python3
"""
Build a wide CSV of NDVI samples per tree and fit one double-logistic curve per tree.

CSV format (date order):
tree_id, DOY1, DOY2, DOY3, ...

Also saves one PNG per tree showing observed NDVI vs DOY and the fitted curve.

Deps:
  pip install geopandas rasterio numpy scipy matplotlib pandas shapely
"""

import os
import re
from glob import glob
from datetime import datetime

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen as rio_sample
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --------------------------- CONFIG ---------------------------
TREE_LAYER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/platanus_x_acerifolia.shp'    # tree points (SHP/GPKG/GeoJSON…)
TREE_LAYER_NAME   = None                     # set if GPKG has multiple layers; else None
NDVI_DIR          = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Planet_ndvi'   # folder with *single-band* NDVI GeoTIFFs
NDVI_GLOB         = "*_ndvi.tif"             # pattern that matches your NDVI files

OUTPUT_DIR        = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope'
WIDE_CSV_NAME     = "ndvi_samples_wide.csv"  # columns: tree_id, DOY1..DOYk (chronological)

AUTO_SCALE_10000  = True   # divide by 10000 if samples look like 0..10000
MIN_OBS_TO_FIT    = 6      # need at least N samples to attempt a fit/plot
SAVE_DOY_MAPPING  = True   # also write a small CSV mapping DOY1.. to actual YYYY-MM-DD & DOY
# --------------------------------------------------------------


# -------- Your double-logistic + weighting (as provided) --------
def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    sigmoid1 = 1 / (1 + np.exp(-mS * (t - S)))
    sigmoid2 = 1 / (1 + np.exp( mA * (t - A)))
    seasonal_term = sigmoid1 + sigmoid2 - 1
    return wNDVI + (mNDVI - wNDVI) * seasonal_term

def weight_function(t, S, A, r):
    tr = 100 * (t - S) / (A - S)
    tr = np.clip(tr, 0, 100)
    return np.exp(-np.abs(r) / (1 + tr / 10))

def fit_curve(t, ndvi_observed):
    # identical structure to your snippet (2-step, residual-weighted refit)
    initial_guess = [np.min(ndvi_observed), np.max(ndvi_observed), np.mean(t), np.mean(t), 1, 1]
    params, _ = curve_fit(double_logistic_function, t, ndvi_observed, p0=initial_guess)
    residuals = ndvi_observed - double_logistic_function(t, *params)
    weights = weight_function(t, params[2], params[3], residuals)
    params, _ = curve_fit(double_logistic_function, t, ndvi_observed, p0=initial_guess, sigma=weights)
    return params
# ---------------------------------------------------------------


# ------------------------ Helpers ------------------------
def parse_date_from_name(path):
    """Parse date from filename: YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD, or YYYYJJJ (year+DOY)."""
    base = os.path.basename(path)

    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try: return datetime(y, mo, d).date()
        except ValueError: pass

    m = re.search(r"(20\d{2})(\d{2})(\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try: return datetime(y, mo, d).date()
        except ValueError: pass

    m = re.search(r"(20\d{2})(\d{3})", base)  # YYYYJJJ
    if m:
        y, jjj = int(m.group(1)), int(m.group(2))
        try: return datetime.strptime(f"{y}{jjj:03d}", "%Y%j").date()
        except ValueError: pass

    # fallback: None (we will skip files we can't date)
    return None

def doy_and_days_in_year(date_obj):
    year = date_obj.year
    doy = int(date_obj.strftime("%j"))
    days = 366 if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) else 365
    return doy, days
# ---------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    wide_csv_path = os.path.join(OUTPUT_DIR, WIDE_CSV_NAME)
    mapping_csv_path = os.path.join(OUTPUT_DIR, "ndvi_wide_columns_mapping.csv")

    # 1) Load trees (ensure points; explode MultiPoint)
    trees = gpd.read_file(TREE_LAYER_PATH, layer=TREE_LAYER_NAME) if TREE_LAYER_NAME else gpd.read_file(TREE_LAYER_PATH)
    if trees.crs is None:
        raise RuntimeError("Tree layer CRS is undefined. Please define/reproject first.")
    trees = trees[trees.geometry.notnull() & trees.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    if "MultiPoint" in trees.geometry.geom_type.unique():
        trees = trees.explode(index_parts=False)
    trees = trees.reset_index(drop=True)

    # 2) Collect NDVI rasters, parse dates, sort CHRONOLOGICALLY
    ndvi_files = sorted(glob(os.path.join(NDVI_DIR, NDVI_GLOB)))
    dated = []
    for f in ndvi_files:
        d = parse_date_from_name(f)
        if d is None:
            continue
        doy, days = doy_and_days_in_year(d)
        dated.append((f, d, doy, days))
    if not dated:
        raise RuntimeError("No NDVI files with parsable dates found. Adjust NDVI_GLOB or date parser.")
    dated.sort(key=lambda x: (x[1], x[0]))  # by calendar date, then name

    # Build column names DOY1..DOYk in this chronological order
    k = len(dated)
    col_names = ["tree_id"] + [f"DOY{i+1}" for i in range(k)]

    # Optional: write a mapping table DOY1.. → actual date & DOY number
    if SAVE_DOY_MAPPING:
        mp_rows = []
        for i, (f, d, doy, _days) in enumerate(dated, start=1):
            mp_rows.append({"col": f"DOY{i}", "date": d.isoformat(), "doy": doy, "file": os.path.basename(f)})
        pd.DataFrame(mp_rows).to_csv(mapping_csv_path, index=False)
        print(f"[OK] Saved DOY column mapping: {mapping_csv_path}")

    # 3) Sample NDVI for each tree across ALL images (in that order)
    # Prepare an empty matrix [n_trees x k] filled with NaN
    n_trees = len(trees)
    wide_vals = np.full((n_trees, k), np.nan, dtype=float)

    # Also keep per-tree observed arrays (t_frac & DOY) for the fit/plot
    per_tree_t = [[] for _ in range(n_trees)]
    per_tree_doy = [[] for _ in range(n_trees)]
    per_tree_ndvi = [[] for _ in range(n_trees)]

    for j, (f, d, doy, days) in enumerate(dated):
        with rasterio.open(f) as src:
            # reproject tree coords if CRS differs
            pts = trees.to_crs(src.crs) if str(trees.crs) != str(src.crs) else trees

            coords = []
            for geom in pts.geometry:
                if geom.geom_type == "MultiPoint":
                    coords.append((geom.geoms[0].x, geom.geoms[0].y) if len(geom.geoms) else (np.nan, np.nan))
                else:
                    coords.append((geom.x, geom.y))

            samples = list(rio_sample(src, coords))
            vals = [float(s[0]) if (s is not None and len(s) > 0) else np.nan for s in samples]

            # auto-scale if NDVI looks like 0..10000
            if AUTO_SCALE_10000:
                try:
                    mx = np.nanmax(vals)
                    if np.isfinite(mx) and mx > 2.0:
                        vals = [v / 10000.0 if np.isfinite(v) else v for v in vals]
                except Exception:
                    pass

            # fill the matrix column j and append to series for plotting/fitting
            t_frac = doy / float(days)  # normalize to [0,1]
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    wide_vals[i, j] = v
                    per_tree_t[i].append(t_frac)
                    per_tree_doy[i].append(doy)
                    per_tree_ndvi[i].append(v)

    # 4) Write the wide CSV (tree_id + DOY1..DOYk)
    rows = []
    for i in range(n_trees):
        row = [i] + [None if np.isnan(wide_vals[i, j]) else float(wide_vals[i, j]) for j in range(k)]
        rows.append(row)
    df = pd.DataFrame(rows, columns=col_names)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(wide_csv_path, index=False)
    print(f"[OK] Saved wide CSV: {wide_csv_path}")

    # 5) Fit ONE double-logistic curve per tree (using your code) and save plot
    for tid in range(n_trees):
        t = np.asarray(per_tree_t[tid], dtype=float)
        y = np.asarray(per_tree_ndvi[tid], dtype=float)
        x_doy = np.asarray(per_tree_doy[tid], dtype=float)

        m = np.isfinite(t) & np.isfinite(y) & np.isfinite(x_doy)
        t, y, x_doy = t[m], y[m], x_doy[m]

        if len(y) < MIN_OBS_TO_FIT:
            continue

        try:
            params = fit_curve(t, y)
        except Exception:
            # If fit fails, skip plotting for this tree
            continue

        # Make dense curve using the same plotting convention as your snippet
        t_fit = np.linspace(float(np.min(t)), float(np.max(t)), 1000)
        ndvi_fit = double_logistic_function(t_fit, *params)

        plt.figure(figsize=(7.5, 4.8))
        # Observed points
        plt.scatter(x_doy, y, label="Observed NDVI")
        # Fitted curve (map t_fit back to ~DOY scale: multiply by 365 to match your example)
        plt.plot(t_fit * 365.0, ndvi_fit, label="Fitted Curve", color="red")
        plt.xlabel("Day of the Year")
        plt.ylabel("NDVI")
        plt.legend()
        plt.title(f"Double Logistic Fit — Tree {tid} (n={len(y)})")
        plt.tight_layout()
        out_png = os.path.join(plot_dir, f"tree_{tid:04d}_ndvi_fit.png")
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[Saved] {out_png}")

    print("[Done]")

if __name__ == "__main__":
    main()
