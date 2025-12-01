#!/usr/bin/env python3
"""
NDVI sampling + robust double-logistic fit per tree + rich phenology metrics,
plus PM2.5 point sample and Impervious/Temperature buffers.

Outputs
-------
- <OUTPUT_DIR>/ndvi_samples_wide.csv
- <OUTPUT_DIR>/ndvi_wide_columns_mapping.csv
- <OUTPUT_DIR>/ndvi_metrics.csv  (tree_id, timings, magnitudes, rates, integrals, fit stats, stressors)
- <OUTPUT_DIR>/plots/tree_<id>_ndvi_fit.png
"""

import os
import re
from glob import glob
from datetime import datetime

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen as rio_sample
from rasterio.windows import Window
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from rasterio.errors import WindowError


# --------------------------- CONFIG ---------------------------
TREE_LAYER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/platanus_x_acerifolia.shp'
TREE_LAYER_NAME   = None

NDVI_DIR          = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Planet_ndvi'
NDVI_GLOB         = "*_ndvi.tif"

OUTPUT_DIR        = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope'
WIDE_CSV_NAME     = "ndvi_samples_wide.csv"
METRICS_CSV_NAME  = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics.csv"

# Single-band PM2.5 raster (point sample)
PM25_RASTER_PATH  = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/Raster_pm25_Brussel.gpkg'

# Impervious fraction raster (0..1 or 0..100). Will output 0..1.
IMPERVIOUS_RASTER_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/impervious_out/merged_impervious.tif'
IMPERVIOUS_BUFFERS_M   = [10, 20, 50, 100]  # meters; 0 = point, others = circular mean

# Temperature raster (single band). Values in K or °C are supported.
TEMP_RASTER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/LST landsat composite/20230606_103337.LST.tif'
TEMP_BUFFERS_M     = [50, 100, 200]      # meters; 0 = point sample

AUTO_SCALE_10000  = True
MIN_OBS_TO_FIT    = 6
SAVE_DOY_MAPPING  = True

# Robust fitting
ROBUST_LOSS       = "soft_l1"
ROBUST_F_SCALE    = 1
IRLS_REFINE       = False
IRLS_ITERS        = 3

# Sampling density for metrics/plots
N_FIT_SAMPLES     = 2000
DAYS_FOR_PLOT     = 365.0

# Shadow mask settings
USE_SHADOW_MASKS      = True
SHADOW_MASK_DIR       = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/shadow rasters'# or a different folder if needed
SHADOW_MASK_SUFFIX    = ".tif.tif"  # how masks are named
SHADOW_IS_SHADOW_FUNC = lambda v: v > 0  # tweak depending on your mask coding


# --------------------------------------------------------------


# -------------------- Sampling helpers ------------------------
def _mean_in_circular_window(arr, cx, cy, radius_px, nodata):
    """Mean of arr inside a circle of radius_px around (cx,cy) in pixel coords."""
    h, w = arr.shape
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx)**2 + (yy - cy)**2 <= radius_px**2
    vals = arr[mask]
    if nodata is not None:
        vals = vals[vals != nodata]
    if vals.size == 0:
        return np.nan
    return float(np.nanmean(vals))

def _sample_multi_buffers_generic(raster_path, trees_gdf, buffers_m, value_postproc=None):
    """
    Generic multi-buffer sampler. Returns dict {radius_m: [values per tree]}.
    - value_postproc: optional function v->v to transform values (e.g., Kelvin->Celsius).
    """
    buffers_m = list(sorted(set(float(r) for r in buffers_m)))
    out = {r: [] for r in buffers_m}

    with rasterio.open(raster_path) as src:
        pts = trees_gdf.to_crs(src.crs) if str(trees_gdf.crs) != str(src.crs) else trees_gdf
        px_w = abs(src.transform.a)
        px_h = abs(src.transform.e)
        px_size = (px_w + px_h) / 2.0
        nodata = src.nodata
        rad_px = {r: (0 if r <= 0 else max(1, int(round(r / max(px_size, 1e-9))))) for r in buffers_m}

        # iterate trees
        for geom in pts.geometry:
            if geom.is_empty:
                for r in buffers_m: out[r].append(np.nan)
                continue
            x, y = (geom.geoms[0].x, geom.geoms[0].y) if geom.geom_type == "MultiPoint" and len(geom.geoms) else (geom.x, geom.y)
            if np.isnan(x) or np.isnan(y):
                for r in buffers_m: out[r].append(np.nan)
                continue
            col, row = src.index(x, y)

            col, row = src.index(x, y)

            # If point is completely outside raster, skip all buffers for this tree
            if (col < 0) or (col >= src.width) or (row < 0) or (row >= src.height):
                for r in buffers_m:
                    out[r].append(np.nan)
                continue

            # --- Point sample (r = 0) ---
            v_point = np.nan
            try:
                win_pt = Window(col, row, 1, 1).intersection(Window(0, 0, src.width, src.height))
                if win_pt.width > 0 and win_pt.height > 0:
                    arr_pt = src.read(1, window=win_pt, boundless=True, fill_value=nodata)
                    v_point = float(arr_pt[0, 0])
                    if nodata is not None and v_point == nodata:
                        v_point = np.nan
            except WindowError:
                # Intersection is empty -> leave v_point as NaN
                v_point = np.nan

            # --- Circular buffers ---
            for r in buffers_m:
                rp = rad_px[r]
                if r <= 0 or rp == 0:
                    v = v_point
                else:
                    try:
                        win = Window(col - rp, row - rp, 2 * rp + 1, 2 * rp + 1).intersection(
                            Window(0, 0, src.width, src.height)
                        )
                    except WindowError:
                        win = None

                    if (win is None) or (win.width <= 0 or win.height <= 0):
                        v = np.nan
                    else:
                        arr = src.read(1, window=win, boundless=True, fill_value=nodata)
                        cx = (col - win.col_off)
                        cy = (row - win.row_off)
                        v = _mean_in_circular_window(arr, cx, cy, rp, nodata)

                if value_postproc is not None and np.isfinite(v):
                    v = value_postproc(v)

                out[r].append(float(v) if np.isfinite(v) else np.nan)

                if value_postproc is not None and np.isfinite(v):
                    v = value_postproc(v)

                out[r].append(float(v) if np.isfinite(v) else np.nan)

    return out

def sample_impervious_multi_buffers(raster_path, trees_gdf, buffers_m):
    # If raster is 0..100 %, convert to 0..1
    def to_unit(v): return v/100.0 if v > 1.5 else v
    return _sample_multi_buffers_generic(raster_path, trees_gdf, buffers_m, value_postproc=to_unit)

def sample_temperature_multi_buffers(raster_path, trees_gdf, buffers_m):
    # Auto-convert Kelvin to °C if values look like Kelvin
    def to_celsius(v): return (v - 273.15) if v > 150 else v
    return _sample_multi_buffers_generic(raster_path, trees_gdf, buffers_m, value_postproc=to_celsius)

def sample_pm25_per_tree(pm25_path, trees_gdf):
    """Sample a single-band PM2.5 raster once per tree (point sample)."""
    with rasterio.open(pm25_path) as src:
        pts = trees_gdf.to_crs(src.crs) if str(trees_gdf.crs) != str(src.crs) else trees_gdf
        coords = []
        for geom in pts.geometry:
            if geom.geom_type == "MultiPoint":
                coords.append((geom.geoms[0].x, geom.geoms[0].y) if len(geom.geoms) else (np.nan, np.nan))
            else:
                coords.append((geom.x, geom.y))
        vals = [float(s[0]) if (s is not None and len(s) > 0) else np.nan
                for s in rio_sample(src, coords)]
    return vals
# ---------------------------------------------------------


# ---------------- Double logistic model ----------------
def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    s1 = 1.0 / (1.0 + np.exp(-mS * (t - S)))
    s2 = 1.0 / (1.0 + np.exp( mA * (t - A)))
    return wNDVI + (mNDVI - wNDVI) * (s1 + s2 - 1.0)

def robust_fit_curve(t, ndvi_observed, loss="soft_l1", f_scale=0.05, max_nfev=20000):
    t = np.asarray(t, float); y = np.asarray(ndvi_observed, float)
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

    def _res(p): return double_logistic_function(t, *p) - y

    try:
        res = least_squares(_res, p0, bounds=(lower, upper), loss=loss, f_scale=f_scale, max_nfev=max_nfev)
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
    t = np.asarray(t, float); y = np.asarray(y, float)
    p = np.array(p_start, float)
    lower = np.array([-1.0, -1.0, 0.00, 0.00, 0.01, 0.01], float)
    upper = np.array([ 1.50,  1.50, 1.00, 1.00, 50.0, 50.0], float)

    for _ in range(iters):
        resid = y - double_logistic_function(t, *p)
        w = _tukey_biweight(resid)
        if np.all(w == 0): break
        def _res(pp): return (double_logistic_function(t, *pp) - y) * np.sqrt(w)
        res = least_squares(_res, p, bounds=(lower, upper), loss="linear", max_nfev=8000)
        p = res.x
        if p[3] <= p[2]:
            p[3] = min(0.99, p[2] + 0.05)
    return p
# --------------------------------------------------------


# ------------------------ Phenology utils ----------------
def first_flat_after(t, y1, y2, start_idx, end_idx, frac_slope=0.15, frac_curv=0.10):
    """
    Find the first index in t[start_idx:end_idx] where |slope| is small and curvature is ~0.
    - frac_slope is relative to max(|slope|) in the window.
    - frac_curv  is relative to max(|curvature|) in the window.
    Returns index into the full arrays; falls back to end_idx-1 if none found.
    """
    start_idx = int(np.clip(start_idx, 0, len(t)-1))
    end_idx   = int(np.clip(end_idx,   1, len(t)))
    if end_idx <= start_idx + 1:
        return end_idx - 1

    win = slice(start_idx, end_idx)
    y1w = y1[win]; y2w = y2[win]
    if not (np.any(np.isfinite(y1w)) and np.any(np.isfinite(y2w))):
        return end_idx - 1

    m_slope = float(np.nanmax(np.abs(y1w)))
    m_curv  = float(np.nanmax(np.abs(y2w)))
    if m_slope == 0 or m_curv == 0:
        return end_idx - 1

    slope_thresh = frac_slope * m_slope
    curv_thresh  = frac_curv  * m_curv

    ok = np.where((np.abs(y1w) <= slope_thresh) & (np.abs(y2w) <= curv_thresh))[0]
    return (start_idx + int(ok[0])) if ok.size else (end_idx - 1)

def auc_between(t, y, t0, t1):
    """AUC between t0 and t1 in 'NDVI·days' (trapezoid rule, t in [0,1])."""
    if t1 <= t0:
        return 0.0
    m = (t >= t0) & (t <= t1)
    if not np.any(m):
        return 0.0
    return float(np.trapz(y[m], t[m]) * DAYS_FOR_PLOT)
# ---------------------------------------------------------


# ------------------------ Helpers ------------------------
def parse_date_from_name(path):
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
    return None

def doy_and_days_in_year(date_obj):
    year = date_obj.year
    doy = int(date_obj.strftime("%j"))
    days = 366 if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) else 365
    return doy, days

def get_shadow_mask_path(ndvi_path):
    """
    Derive the corresponding shadow mask path from an NDVI raster path.
    Adjust this to match your naming convention.
    """
    base = os.path.basename(ndvi_path)
    # Example: 2021_06_15_ndvi.tif -> 2021_06_15_shadow.tif
    if "_ndvi" in base:
        shadow_name = "shadow_trees_" + base.replace("_ndvi", ".tif")
    else:
        shadow_name = "shadow_trees_" + base
    return os.path.join(SHADOW_MASK_DIR, shadow_name)

# ---------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    wide_csv_path    = os.path.join(OUTPUT_DIR, WIDE_CSV_NAME)
    mapping_csv_path = os.path.join(OUTPUT_DIR, "ndvi_wide_columns_mapping.csv")
    metrics_csv_path = os.path.join(OUTPUT_DIR, METRICS_CSV_NAME)

    # 1) Load trees
    trees = gpd.read_file(TREE_LAYER_PATH, layer=TREE_LAYER_NAME) if TREE_LAYER_NAME else gpd.read_file(TREE_LAYER_PATH)
    if trees.crs is None:
        raise RuntimeError("Tree layer CRS is undefined.")
    trees = trees[trees.geometry.notnull() & trees.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    if "MultiPoint" in trees.geometry.geom_type.unique():
        trees = trees.explode(index_parts=False)
    trees = trees.reset_index(drop=True)
    print(f"Total number of trees before filtering: {len(trees)}")

    # --- Filter trees by height >= 10 m
    if "height" not in trees.columns:
        raise RuntimeError("Attribute 'height' not found in tree layer.")
    #trees = trees[trees["height"] >= 10].copy()
    #trees = trees.reset_index(drop=True)
    print(f"[INFO] Trees retained after height filter (>=10 m): {len(trees)}")

    # --- Filter trees by area >= 36 m --> this is 4 3x3 pixels
    if "area" not in trees.columns:
        raise RuntimeError("Attribute 'area' not found in tree layer.")
    trees = trees[trees["area"] >= 36].copy()
    trees = trees.reset_index(drop=True)
    print(f"[INFO] Trees retained after area filter (>=36 m): {len(trees)}")

    # --- PM2.5 (point) ---
    pm25_vals = sample_pm25_per_tree(PM25_RASTER_PATH, trees)

    # --- Impervious multi-buffers ---
    imperv_by_r = sample_impervious_multi_buffers(
        IMPERVIOUS_RASTER_PATH,
        trees,
        buffers_m=IMPERVIOUS_BUFFERS_M
    )
    print("[OK] Impervious buffers (m):", IMPERVIOUS_BUFFERS_M)

    # --- Temperature multi-buffers (°C) ---
    temp_by_r = sample_temperature_multi_buffers(
        TEMP_RASTER_PATH, trees, buffers_m=TEMP_BUFFERS_M
    )
    print("[OK] Temperature buffers (m):", TEMP_BUFFERS_M)

    # 2) NDVI rasters + dates
    ndvi_files = sorted(glob(os.path.join(NDVI_DIR, NDVI_GLOB)))
    dated = []
    for f in ndvi_files:
        d = parse_date_from_name(f)
        if d is None:
            continue
        doy, days = doy_and_days_in_year(d)
        dated.append((f, d, doy, days))
    if not dated:
        raise RuntimeError("No NDVI files with parsable dates found.")
    dated.sort(key=lambda x: (x[1], x[0]))

    # Wide CSV columns
    k = len(dated)
    col_names = ["tree_id"] + [f"DOY{i+1}" for i in range(k)]
    if SAVE_DOY_MAPPING:
        mp_rows = [{"col": f"DOY{i+1}", "date": d.isoformat(), "doy": doy, "file": os.path.basename(f)}
                   for i, (f, d, doy, _days) in enumerate(dated)]
        pd.DataFrame(mp_rows).to_csv(mapping_csv_path, index=False)
        print(f"[OK] Saved DOY column mapping: {mapping_csv_path}")

    shadow_flag_tree = np.zeros(len(trees), dtype=bool)

    # 3) Sample NDVI per date (point sample)
    n_trees = len(trees)
    wide_vals = np.full((n_trees, k), np.nan, dtype=float)
    per_tree_t, per_tree_doy, per_tree_ndvi = [[] for _ in range(n_trees)], [[] for _ in range(n_trees)], [[] for _ in range(n_trees)]

    for j, (f, d, doy, days) in enumerate(dated):
        with rasterio.open(f) as src:
            pts = trees.to_crs(src.crs) if str(trees.crs) != str(src.crs) else trees
            coords = []
            for geom in pts.geometry:
                if geom.geom_type == "MultiPoint":
                    coords.append((geom.geoms[0].x, geom.geoms[0].y) if len(geom.geoms) else (np.nan, np.nan))
                else:
                    coords.append((geom.x, geom.y))

            # --- Sample NDVI ---
            samples = list(rio_sample(src, coords))
            vals = [float(s[0]) if (s is not None and len(s) > 0) else np.nan for s in samples]

            # --- Auto-scale NDVI if needed ---
            if AUTO_SCALE_10000:
                try:
                    mx = np.nanmax(vals)
                    if np.isfinite(mx) and mx > 2.0:
                        vals = [v / 10000.0 if np.isfinite(v) else v for v in vals]
                except Exception:
                    pass

            # --- Sample shadow mask (optional) ---
            shadow_flags = [False] * len(vals)
            if USE_SHADOW_MASKS:
                shadow_path = get_shadow_mask_path(f)
                if os.path.exists(shadow_path):
                    try:
                        with rasterio.open(shadow_path) as sm:
                            # Assume same grid; reproject if needed
                            pts_shadow = pts.to_crs(sm.crs) if str(pts.crs) != str(sm.crs) else pts
                            coords_shadow = []
                            for geom in pts_shadow.geometry:
                                if geom.geom_type == "MultiPoint":
                                    coords_shadow.append(
                                        (geom.geoms[0].x, geom.geoms[0].y) if len(geom.geoms) else (np.nan, np.nan))
                                else:
                                    coords_shadow.append((geom.x, geom.y))
                            shadow_samples = list(rio_sample(sm, coords_shadow))
                            shadow_vals = [float(s[0]) if (s is not None and len(s) > 0) else np.nan for s in
                                           shadow_samples]

                            # Decide which trees are in shadow for this date
                            shadow_flags = [
                                (np.isfinite(v) and SHADOW_IS_SHADOW_FUNC(v))
                                for v in shadow_vals
                            ]
                    except Exception as e:
                        print(f"[WARN] Could not use shadow mask for {f}: {e}")
                else:
                    print(f"[WARN] Shadow mask not found for {f}: expected {shadow_path}")

            # --- Store NDVI only if NOT in shadow ---
            # If tree is shadowed on this date → mark tree as invalid permanently
            for i in range(len(shadow_flags)):
                if shadow_flags[i]:
                    shadow_flag_tree[i] = True


            t_frac = doy / float(days)
            for i, v in enumerate(vals):
                if shadow_flag_tree[i]:
                    continue  # skip the entire profile for this tree

                if np.isfinite(v):
                    wide_vals[i, j] = v
                    per_tree_t[i].append(t_frac)
                    per_tree_doy[i].append(doy)
                    per_tree_ndvi[i].append(v)

    # 4) Write wide CSV
    df_wide = pd.DataFrame([[i] + [None if np.isnan(wide_vals[i, j]) else float(wide_vals[i, j]) for j in range(k)]
                            for i in range(n_trees)], columns=col_names)
    df_wide.to_csv(wide_csv_path, index=False)
    print(f"[OK] Saved wide CSV: {wide_csv_path}")

    # 5) Fit + metrics + plots
    metrics_rows = []
    id_field = None
    for cand in ["crown_id", "tree_id", "id", "ID"]:
        if cand in trees.columns:
            id_field = cand; break

    # Pre-name stressor columns
    imperv_cols = {r: f"impervious_r{int(r) if float(r).is_integer() else r}" for r in IMPERVIOUS_BUFFERS_M}
    temp_cols   = {r: f"temp_r{int(r) if float(r).is_integer() else r}"       for r in TEMP_BUFFERS_M}

    for tid in range(n_trees):
        t = np.asarray(per_tree_t[tid], dtype=float)
        y = np.asarray(per_tree_ndvi[tid], dtype=float)
        x_doy = np.asarray(per_tree_doy[tid], dtype=float)
        m = np.isfinite(t) & np.isfinite(y) & np.isfinite(x_doy)
        t, y, x_doy = t[m], y[m], x_doy[m]
        if len(y) < MIN_OBS_TO_FIT:
            continue

        p = robust_fit_curve(t, y, loss=ROBUST_LOSS, f_scale=ROBUST_F_SCALE)
        if p is None:
            continue
        if IRLS_REFINE:
            p = irls_refine(t, y, p, iters=IRLS_ITERS)

        # Evaluate smooth curve & derivatives
        t_fit = np.linspace(0.0, 1.0, N_FIT_SAMPLES)
        y_fit = double_logistic_function(t_fit, *p)
        y1 = np.gradient(y_fit, t_fit)          # slope
        y2 = np.gradient(y1, t_fit)             # curvature
        y3 = np.gradient(y2, t_fit)             # third derivative

        # Params & basic magnitudes
        wNDVI, mNDVI, S_est, A_est, mS_est, mA_est = map(float, p)
        ndvi_base = float(wNDVI)
        ndvi_max  = float(mNDVI)
        amplitude = float(ndvi_max - ndvi_base)

        # --- SOS (3rd-derivative peak in early window) ---
        t0 = max(0.0, S_est - 0.20)
        t1 = min(1.0, (S_est + A_est) / 2.0)
        mask = (t_fit >= t0) & (t_fit <= t1)
        if not np.any(mask):
            mask = (t_fit >= 0.0) & (t_fit <= max(0.5, S_est))
        idx_sos_local = int(np.argmax(y3[mask]))
        idx_mask = np.where(mask)[0]
        idx_sos = int(idx_mask[idx_sos_local])
        sos_t   = float(t_fit[idx_sos])
        sos_doy = sos_t * DAYS_FOR_PLOT
        ndvi_sos = float(np.interp(sos_t, t_fit, y_fit))

        # --- POS (flattening after steepest ascent) ---
        post_end = min(1.0, S_est + A_est + 0.10)
        post_mask = (t_fit > sos_t) & (t_fit <= post_end)
        if not np.any(post_mask):
            post_mask = (t_fit > sos_t)
        i_post = np.where(post_mask)[0]
        i_maxslope = i_post[int(np.nanargmax(y1[i_post]))] if i_post.size else int(np.nanargmax(y1))

        peak_idx = first_flat_after(t_fit, y1, y2, i_maxslope, len(t_fit), frac_slope=0.15, frac_curv=0.10)
        peak_t = float(t_fit[peak_idx])
        peak_doy = peak_t * DAYS_FOR_PLOT
        ndvi_peak = float(np.interp(peak_t, t_fit, y_fit))

        # --- Senescence onset (SO): first *meaningfully negative* slope after peak ---
        dec_start = peak_idx
        dec_end = len(t_fit)
        i_range = np.arange(dec_start, dec_end)

        # Steepest negative slope after peak (reference magnitude)
        min_slope_after_peak = float(np.nanmin(y1[i_range])) if i_range.size else float(np.nanmin(y1))
        decline_mag = abs(min_slope_after_peak)  # >0

        # Threshold for "meaningfully negative": fraction of steepest decline
        theta = 0.15  # 10–20% works well; tune if needed
        # Candidates where slope <= -theta*decline_mag (and optionally curvature concave-down)
        cand_onset = np.where((y1[i_range] <= -theta * decline_mag) & (y2[i_range] <= 0))[0]

        if cand_onset.size:
            onset_idx = int(i_range[cand_onset[0]])
        else:
            # Fallbacks: first zero-crossing of slope, else steepest decline
            zc = np.where(y1[i_range] <= 0)[0]
            onset_idx = int(i_range[zc[0]]) if zc.size else int(
                i_range[int(np.nanargmin(y1[i_range]))]) if i_range.size else peak_idx

        onset_t = float(t_fit[onset_idx])
        onset_doy = onset_t * DAYS_FOR_PLOT
        ndvi_onset = float(np.interp(onset_t, t_fit, y_fit))

        # --- EOS (flattening on decline, as before: first flat after steepest decline) ---
        i_min_slope = int(i_range[int(np.nanargmin(y1[i_range]))]) if i_range.size else dec_start
        eos_idx = first_flat_after(t_fit, y1, y2, i_min_slope, dec_end, frac_slope=0.15, frac_curv=0.10)
        eos_t = float(t_fit[eos_idx])
        eos_doy = eos_t * DAYS_FOR_PLOT
        ndvi_eos = float(np.interp(eos_t, t_fit, y_fit))
        # With this definition, EOS ≈ "complete leaf senescence" (flattened decline)

        # Durations / asymmetry (update with onset)
        los_days = float(max(0.0, eos_doy - sos_doy))
        greenup_days = float(max(0.0, peak_doy - sos_doy))
        plateau_days = float(max(0.0, onset_doy - peak_doy))  # flat-top/plateau length
        decline_days = float(max(0.0, eos_doy - onset_doy))  # actual senescence duration
        asymmetry = float((decline_days - greenup_days) / los_days) if los_days > 0 else np.nan

        # Rates (keep your green-up; add mean decline rate onset→eos if desired)
        dt_days = max(1e-6, peak_doy - sos_doy)
        slope_sos_peak = (ndvi_peak - ndvi_sos) / dt_days
        senescence_rate = min_slope_after_peak * DAYS_FOR_PLOT  # steepest (negative)
        mean_senescence_rate = (ndvi_eos - ndvi_onset) / max(1e-6, decline_days)  # average decline

        # --- Integrals (AUC) ---
        auc_full = float(np.trapz(y_fit, t_fit) * DAYS_FOR_PLOT)           # NDVI·days
        auc_sos_eos = auc_between(t_fit, y_fit, sos_t, eos_t)
        auc_above_base_full = float(np.trapz(np.maximum(0.0, y_fit - ndvi_base), t_fit) * DAYS_FOR_PLOT)
        auc_above_base_sos_eos = auc_between(t_fit, np.maximum(0.0, y_fit - ndvi_base), sos_t, eos_t)

        # --- Fit residual quality (using observed points) ---
        y_hat_obs = double_logistic_function(t, *p)
        resid = y - y_hat_obs
        rmse  = float(np.sqrt(np.nanmean((resid)**2))) if resid.size else np.nan
        iqr   = float(np.nanpercentile(resid, 75) - np.nanpercentile(resid, 25)) if resid.size else np.nan

        # --- Assemble row ---
        row = {
            "tree_id": trees[id_field].iloc[tid] if id_field else tid,
            # timings
            "sos_doy": round(sos_doy, 3),
            "peak_doy": round(peak_doy, 3),
            "eos_doy": round(eos_doy, 3),
            "los_days": round(los_days, 3),
            "sen_onset_doy": round(onset_doy, 3),
            # magnitudes
            "ndvi_base": round(ndvi_base, 6),
            "ndvi_peak": round(ndvi_peak, 6),
            "ndvi_eos":  round(ndvi_eos, 6),
            "amplitude": round(amplitude, 6),
            "ndvi_sen_onset": round(ndvi_onset, 6),
            # durations
            "plateau_days": round(plateau_days, 3),
            "decline_days": round(decline_days, 3),
            # rates / shapes
            "slope_sos_peak": round(slope_sos_peak, 9),
            "senescence_rate": round(senescence_rate, 9),   # NDVI/day (negative)
            "asymmetry": round(asymmetry, 6),
            "mean_senescence_rate": round(mean_senescence_rate, 9),
            # integrals
            "auc_full": round(auc_full, 3),
            "auc_sos_eos": round(auc_sos_eos, 3),
            "auc_above_base_full": round(auc_above_base_full, 3),
            "auc_above_base_sos_eos": round(auc_above_base_sos_eos, 3),
            # fit quality
            "rmse": round(rmse, 6),
            "resid_iqr": round(iqr, 6),
            # stressors
            "pm25": pm25_vals[tid] if (tid < len(pm25_vals) and pm25_vals[tid] is not None) else np.nan,
        }

        # attach impervious per radius
        for r in IMPERVIOUS_BUFFERS_M:
            vals = imperv_by_r[r]
            row[imperv_cols[r]] = vals[tid] if tid < len(vals) else np.nan

        # attach temperature per radius (°C)
        for r in TEMP_BUFFERS_M:
            vals = temp_by_r[r]
            row[temp_cols[r]] = vals[tid] if tid < len(vals) else np.nan

        metrics_rows.append(row)

        # Plot
        plt.figure(figsize=(7.8, 4.8))
        plt.scatter(x_doy, y, s=22, alpha=0.9, label="Observed NDVI")
        plt.plot(t_fit * DAYS_FOR_PLOT, y_fit, linewidth=2, label="Robust fit")
        plt.axvline(sos_doy, linestyle="--", linewidth=1.5, label=f"SOS ≈ {sos_doy:.1f}")
        plt.axvline(peak_doy, linestyle="--", linewidth=1.5, label=f"Peak ≈ {peak_doy:.1f}")
        plt.axvline(eos_doy, linestyle="--", linewidth=1.5, label=f"EOS ≈ {eos_doy:.1f}")
        plt.axvline(onset_doy, linestyle="--", linewidth=1.5, label=f"SO ≈ {onset_doy:.1f}")
        plt.xlabel("Day of Year"); plt.ylabel("NDVI")
        plt.title(f"Tree {row['tree_id']} — SOS/POS/EOS + metrics")
        plt.grid(True, alpha=0.3); plt.legend()
        out_png = os.path.join(plot_dir, f"tree_{tid:04d}_ndvi_fit.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
        print(f"[Saved] {out_png}")

    # Save metrics CSV
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(METRICS_CSV_NAME, index=False)
    print(f"[OK] Saved metrics CSV with stressor columns "
          f"{list(imperv_cols.values()) + list(temp_cols.values())}: {METRICS_CSV_NAME}")
    print("[Done]")


if __name__ == "__main__":
    main()
