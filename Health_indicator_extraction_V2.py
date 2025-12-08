#!/usr/bin/env python3
"""
NDVI sampling + robust double-logistic fit per tree + rich phenology metrics,
plus PM2.5 point sample and Impervious/Temperature buffers.

Shadow-based filtering:
- For each tree, we also sample tree-top shade (shadow rasters) per date.
- If a tree has more than MAX_SHADE_POINTS shaded NDVI observations, its entire
  NDVI profile is discarded (no fit, no metrics, no plot).

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

from scipy.optimize import least_squares
from scipy.signal import find_peaks


# --------------------------- CONFIG ---------------------------
OUTPUT_DIR        = ('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/acer platanoides')

TREE_LAYER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/Acer_platanoides.shp'
TREE_LAYER_NAME   = None
OUTPUT_TREE_LAYER_PATH = os.path.join(OUTPUT_DIR, "trees_with_pheno.gpkg")


NDVI_DIR          = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Planet_ndvi'
NDVI_GLOB         = "*_ndvi.tif"

WIDE_CSV_NAME     = "ndvi_samples_wide.csv"
METRICS_CSV_NAME  = OUTPUT_DIR + "/ndvi_metrics.csv"

# Tree-top shadow rasters (one per composite) from the other script
SHADOW_DIR        = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/shadow rasters'
SHADOW_PREFIX     = "shadow_trees_"     # produced as f"shadow_trees_{image_id}.tif"
SHADOW_SUFFIX     = ".tif"

# Max number of shaded NDVI observations allowed per tree before discarding its profile
MAX_SHADE_POINTS  = 3

# Single-band PM2.5 raster (point sample)
PM25_RASTER_PATH  = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/Raster_pm25_Brussel.gpkg'

# Folder with multiple pollution rasters (each single-band)
POLLUTION_DIR   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/pollution'
POLLUTION_GLOB  = '*.tif'   # adapt if needed (e.g. '*.gpkg' or specific prefix)

# Impervious fraction raster (0..1 or 0..100). Will output 0..1.
IMPERVIOUS_RASTER_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/impervious_out/merged_impervious.tif'
IMPERVIOUS_BUFFERS_M   = [10, 20, 50, 100]  # meters; 0 = point, others = circular mean

# Temperature raster (single band). Values in K or °C are supported.
TEMP_RASTER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/Temperature_layers_Gaëlle/max_Brussels_annual.tif'
TEMP_BUFFERS_M     = [100, 200]      # meters; 0 = point sample

# Temperature raster from LST (single band). Values in K or °C are supported.
LST_TEMP_RASTER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/LST landsat composite/20230614_103335_ST_B10_lambert.tif'
LST_TEMP_BUFFERS_M     = [50, 100, 200]      # meters; 0 = point sample

# Insolation raster
INSOLATION_RASTER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/insol_gs.tif'
INS_BUFFERS_M     = [3, 9]      # meters; 0 = point sample

AUTO_SCALE_10000  = True
MIN_OBS_TO_FIT    = 6
SAVE_DOY_MAPPING  = True

# Robust fitting
ROBUST_LOSS       = "soft_l1"
ROBUST_F_SCALE    = 0.2
IRLS_REFINE       = True
IRLS_ITERS        = 3

# Sampling density for metrics/plots
N_FIT_SAMPLES     = 2000
DAYS_FOR_PLOT     = 365.0
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

def sample_singleband_raster_per_tree(raster_path, trees_gdf):
    """Sample a single-band raster once per tree (point sample)."""
    with rasterio.open(raster_path) as src:
        pts = trees_gdf.to_crs(src.crs) if str(trees_gdf.crs) != str(src.crs) else trees_gdf
        coords = []
        for geom in pts.geometry:
            if geom.geom_type == "MultiPoint":
                coords.append((geom.geoms[0].x, geom.geoms[0].y) if len(geom.geoms) else (np.nan, np.nan))
            else:
                coords.append((geom.x, geom.y))
        vals = [
            float(s[0]) if (s is not None and len(s) > 0) else np.nan
            for s in rio_sample(src, coords)
        ]
    return vals

def sample_pollution_folder(pollution_dir, pattern, trees_gdf):
    """
    Sample all single-band pollution rasters in a folder.
    Returns dict: {short_name: [values per tree]}.
    short_name = filename without extension, cleaned a bit for column names.
    """
    poll_files = sorted(glob(os.path.join(pollution_dir, pattern)))
    out = {}
    for path in poll_files:
        base = os.path.basename(path)
        short = os.path.splitext(base)[0]
        # Optional: make sure column-name-friendly
        short = re.sub(r'\W+', '_', short)
        print(f"[POLL] Sampling {base}")
        out[short] = sample_singleband_raster_per_tree(path, trees_gdf)
    return out

def shade_path_from_ndvi(ndvi_path):
    """
    Given NDVI raster path like 'COMPOSITEID_ndvi.tif',
    return expected shadow raster path 'SHADOW_DIR/shadow_trees_COMPOSITEID.tif'.
    """
    base = os.path.basename(ndvi_path)
    m = re.search(r"(.+?)_ndvi\.tif$", base)
    if not m:
        return None
    image_id = m.group(1)
    return os.path.join(SHADOW_DIR, f"{SHADOW_PREFIX}{image_id}{SHADOW_SUFFIX}")
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

def find_pheno_events_from_y3(t_fit, y3):
    """
    Detect phenology events from the 3rd derivative y3(t):

      greenup    : 1st positive peak in y3
      sos        : 1st pit (negative peak) after greenup
      maturity   : 2nd positive peak after sos
      senescence : 2nd pit after maturity
      eos        : 3rd positive peak after senescence
      dormancy   : 3rd pit after eos

    Returns dict with indices (into t_fit) for:
      ['greenup', 'sos', 'maturity', 'senescence', 'eos', 'dormancy']
    """
    t_fit = np.asarray(t_fit, float)
    y3 = np.asarray(y3, float)
    n = len(t_fit)

    events = {
        "greenup": None,
        "sos": None,
        "maturity": None,
        "senescence": None,
        "eos": None,
        "dormancy": None,
    }

    if n == 0 or not np.any(np.isfinite(y3)):
        for k in events:
            events[k] = 0
        return events

    y3_clean = np.where(np.isfinite(y3), y3, 0.0)
    max_abs = float(np.nanmax(np.abs(y3_clean)))
    if not np.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1.0
    prom = 0.05 * max_abs  # 5% of max magnitude to suppress tiny wiggles
    if prom <= 0:
        prom = 1e-6

    # Positive peaks in y3
    peaks, _ = find_peaks(y3_clean, prominence=prom)
    # Negative peaks (pits) in y3
    pits, _ = find_peaks(-y3_clean, prominence=prom)

    peaks = np.array(sorted(peaks))
    pits  = np.array(sorted(pits))

    def first_after(indices, after_idx):
        """
        Return the first element of `indices` that is strictly > after_idx.
        If after_idx is None, simply return the first index.
        If none found, return None.
        """
        if indices is None or len(indices) == 0:
            return None

        # If no previous event exists → just return the first peak/pit
        if after_idx is None:
            return int(indices[0])

        # Normal case: find first value > after_idx
        for idx in indices:
            if idx > after_idx:
                return int(idx)

        return None  # none found

    # 1) Green-up: 1st peak (or global max if none)
    if peaks.size > 0:
        gu = int(peaks[0])
    else:
        gu = int(np.nanargmax(y3_clean))
    events["greenup"] = gu

    # 2) SOS: 1st pit after green-up
    sos = first_after(pits, gu) if pits.size > 0 else None
    events["sos"] = sos

    # 3) Maturity: 2nd peak (after SOS)
    if sos is not None:
        mat = first_after(peaks, sos)
        if mat is None and peaks.size > 1:
            mat = int(peaks[1])
    else:
        mat = int(peaks[1]) if peaks.size > 1 else gu
    events["maturity"] = mat

    # 4) Senescence: 2nd pit (after maturity)
    sen = first_after(pits, mat) if pits.size > 0 else None
    if sen is None and pits.size > 1:
        sen = int(pits[1])
    events["senescence"] = sen

    # 5) EOS: 3rd peak (after senescence)
    eos = first_after(peaks, sen if sen is not None else mat)
    if eos is None:
        if peaks.size >= 3:
            eos = int(peaks[2])
        elif peaks.size > 0:
            eos = int(peaks[-1])
        else:
            eos = n - 1
    events["eos"] = eos

    # 6) Dormancy: 3rd pit (after EOS)
    dorm = first_after(pits, eos if eos is not None else (n - 1))
    if dorm is None:
        if pits.size >= 3:
            dorm = int(pits[2])
        elif pits.size > 0:
            dorm = int(pits[-1])
        else:
            dorm = n - 1
    events["dormancy"] = dorm

    # Make sure they’re defined and non-decreasing
    last = 0
    for key in ["greenup", "sos", "maturity", "senescence", "eos", "dormancy"]:
        if events[key] is None:
            events[key] = last
        else:
            events[key] = int(np.clip(events[key], last, n - 1))
        last = events[key]

    return events

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
    # trees = trees[trees["height"] >= 10].copy()
    # trees = trees.reset_index(drop=True)
    print(f"[INFO] Trees retained after height filter (>=10 m): {len(trees)}")

    # --- Filter trees by area >= 36 m --> this is 4 3x3 pixels
    if "area" not in trees.columns:
        raise RuntimeError("Attribute 'area' not found in tree layer.")
    trees = trees[trees["area"] >= 36].copy()
    trees = trees.reset_index(drop=True)
    print(f"[INFO] Trees retained after area filter (>=36 m): {len(trees)}")

    # --- Pollution rasters (point samples) ---
    pollution_by_layer = sample_pollution_folder(
        POLLUTION_DIR,
        POLLUTION_GLOB,
        trees
    )
    print("[OK] Pollution layers sampled:", list(pollution_by_layer.keys()))

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

    # --- LST Temperature multi-buffers (°C) ---
    LST_temp_by_r = sample_temperature_multi_buffers(
        LST_TEMP_RASTER_PATH, trees, buffers_m=LST_TEMP_BUFFERS_M
    )
    print("[OK] Temperature buffers (m):", TEMP_BUFFERS_M)

    # --- LST Temperature multi-buffers (°C) ---
    insolation_by_r = sample_temperature_multi_buffers(
        INSOLATION_RASTER_PATH, trees, buffers_m=INS_BUFFERS_M
    )
    print("[OK] insolation buffers (m):", INS_BUFFERS_M)

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

    # 3) Sample NDVI per date (point sample) + shadow flags
    n_trees = len(trees)
    wide_vals = np.full((n_trees, k), np.nan, dtype=float)
    per_tree_t      = [[] for _ in range(n_trees)]
    per_tree_doy    = [[] for _ in range(n_trees)]
    per_tree_ndvi   = [[] for _ in range(n_trees)]
    per_tree_shade  = [[] for _ in range(n_trees)]  # 0/1 per NDVI observation

    warned_no_shadow = False

    for j, (f, d, doy, days) in enumerate(dated):
        with rasterio.open(f) as src:
            pts = trees.to_crs(src.crs) if str(trees.crs) != str(src.crs) else trees
            coords = []
            for geom in pts.geometry:
                if geom.geom_type == "MultiPoint":
                    coords.append((geom.geoms[0].x, geom.geoms[0].y) if len(geom.geoms) else (np.nan, np.nan))
                else:
                    coords.append((geom.x, geom.y))
            ndvi_samples = list(rio_sample(src, coords))
            ndvi_vals = [float(s[0]) if (s is not None and len(s) > 0) else np.nan for s in ndvi_samples]

        # Shadow sampling for the same composite (tree-top shade)
        shade_path = shade_path_from_ndvi(f)
        shade_vals = None
        if shade_path is not None and os.path.exists(shade_path):
            with rasterio.open(shade_path) as shade_src:
                # Assume same CRS / grid as NDVI (produced from same Planet composite)
                shade_samples = list(rio_sample(shade_src, coords))
                shade_vals = [float(s[0]) if (s is not None and len(s) > 0) else np.nan
                              for s in shade_samples]
        else:
            if not warned_no_shadow:
                print(f"[WARN] No shadow raster found for NDVI file {os.path.basename(f)} "
                      f"(expected {shade_path}). Treating all as unshaded for this file (and any similar).")
                warned_no_shadow = True

        if AUTO_SCALE_10000:
            try:
                mx = np.nanmax(ndvi_vals)
                if np.isfinite(mx) and mx > 2.0:
                    ndvi_vals = [v / 10000.0 if np.isfinite(v) else v for v in ndvi_vals]
            except Exception:
                pass

        t_frac = doy / float(days)  # [0,1]
        for i, v in enumerate(ndvi_vals):
            if np.isfinite(v):
                wide_vals[i, j] = v
                per_tree_t[i].append(t_frac)
                per_tree_doy[i].append(doy)
                per_tree_ndvi[i].append(v)

                # Shade flag for this observation
                s_flag = 0
                if shade_vals is not None and i < len(shade_vals):
                    sv = shade_vals[i]
                    if np.isfinite(sv) and sv > 0.5:
                        s_flag = 1
                per_tree_shade[i].append(s_flag)

    # 4) Write wide CSV (all NDVI samples, before shade-based filtering)
    df_wide = pd.DataFrame(
        [[i] + [None if np.isnan(wide_vals[i, j]) else float(wide_vals[i, j]) for j in range(k)]
         for i in range(n_trees)],
        columns=col_names
    )

    # Attach tree point attributes
    df_wide["height"] = trees["height"].values
    df_wide["area"] = trees["area"].values

    df_wide.to_csv(wide_csv_path, index=False)
    print(f"[OK] Saved wide CSV: {wide_csv_path}")

    # 5) Fit + metrics + plots (with shade-profile filtering)
    metrics_rows = []
    id_field = None
    for cand in ["crown_id", "tree_id", "id", "ID"]:
        if cand in trees.columns:
            id_field = cand; break

    # Pre-name stressor columns
    imperv_cols = {r: f"impervious_r{int(r) if float(r).is_integer() else r}" for r in IMPERVIOUS_BUFFERS_M}
    temp_cols   = {r: f"temp_r{int(r) if float(r).is_integer() else r}"       for r in TEMP_BUFFERS_M}
    LST_temp_cols = {r: f"lst_temp_r{int(r) if float(r).is_integer() else r}" for r in LST_TEMP_BUFFERS_M}
    insolation_cols = {r: f"insolation{int(r) if float(r).is_integer() else r}" for r in INS_BUFFERS_M}

    for tid in range(n_trees):
        t = np.asarray(per_tree_t[tid], dtype=float)
        y = np.asarray(per_tree_ndvi[tid], dtype=float)
        x_doy = np.asarray(per_tree_doy[tid], dtype=float)
        shade_flags = np.asarray(per_tree_shade[tid], dtype=float) if per_tree_shade[tid] else np.array([], float)

        m = np.isfinite(t) & np.isfinite(y) & np.isfinite(x_doy)
        t, y, x_doy = t[m], y[m], x_doy[m]
        if shade_flags.size == m.size:
            shade_flags = shade_flags[m]
        # If shade_flags is shorter for some reason, just ignore mask
        if shade_flags.size != t.size:
            shade_flags = shade_flags[:t.size]

        if len(y) < MIN_OBS_TO_FIT:
            continue

        # --- Shade-based temporal filter: remove trees with too many shaded NDVI points ---
        if shade_flags.size:
            shade_count = int(np.nansum(shade_flags > 0.5))
        else:
            shade_count = 0

        if shade_count > MAX_SHADE_POINTS:
            # Skip this tree entirely
            print(f"[FILTER] Tree index {tid} (ID={trees[id_field].iloc[tid] if id_field else tid}) "
                  f"has {shade_count} shaded NDVI observations (> {MAX_SHADE_POINTS}); skipping.")
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

        # --- Phenology events from 3rd derivative (y3) ---
        ev = find_pheno_events_from_y3(t_fit, y3)
        gu_idx        = ev["greenup"]
        sos_idx       = ev["sos"]
        mat_idx       = ev["maturity"]
        sen_idx       = ev["senescence"]
        eos_idx       = ev["eos"]
        dorm_idx      = ev["dormancy"]

        # Convert to t and DOY
        gu_t    = float(t_fit[gu_idx]);    gu_doy    = gu_t * DAYS_FOR_PLOT
        sos_t   = float(t_fit[sos_idx]);   sos_doy   = sos_t * DAYS_FOR_PLOT
        peak_t  = float(t_fit[mat_idx]);   peak_doy  = peak_t * DAYS_FOR_PLOT
        onset_t = float(t_fit[sen_idx]);   onset_doy = onset_t * DAYS_FOR_PLOT
        eos_t   = float(t_fit[eos_idx]);   eos_doy   = eos_t * DAYS_FOR_PLOT
        dorm_t  = float(t_fit[dorm_idx]);  dorm_doy  = dorm_t * DAYS_FOR_PLOT

        # NDVI at those times
        ndvi_gu    = float(np.interp(gu_t,   t_fit, y_fit))
        ndvi_sos   = float(np.interp(sos_t,  t_fit, y_fit))
        ndvi_peak  = float(np.interp(peak_t, t_fit, y_fit))
        ndvi_onset = float(np.interp(onset_t,t_fit, y_fit))
        ndvi_eos   = float(np.interp(eos_t,  t_fit, y_fit))
        ndvi_dorm  = float(np.interp(dorm_t, t_fit, y_fit))


        # Durations / asymmetry
        los_days      = float(max(0.0, eos_doy - sos_doy))
        greenup_days  = float(max(0.0, peak_doy - sos_doy))
        plateau_days  = float(max(0.0, onset_doy - peak_doy))
        decline_days  = float(max(0.0, eos_doy - onset_doy))
        asymmetry     = float((decline_days - greenup_days) / los_days) if los_days > 0 else np.nan

        # Rates
        dt_days = max(1e-6, peak_doy - sos_doy)
        slope_sos_peak = (ndvi_peak - ndvi_sos) / dt_days

        # Decline rates (using first derivative)
        dec_start = sen_idx
        dec_end   = len(t_fit)
        i_range   = np.arange(dec_start, dec_end)
        min_slope_after_onset = float(np.nanmin(y1[i_range])) if i_range.size else float(np.nanmin(y1))
        decline_mag = abs(min_slope_after_onset)

        senescence_rate      = min_slope_after_onset * DAYS_FOR_PLOT
        mean_senescence_rate = (ndvi_eos - ndvi_onset) / max(1e-6, decline_days)

        # --- Integrals (AUC) ---
        auc_full = float(np.trapz(y_fit, t_fit) * DAYS_FOR_PLOT)
        auc_sos_eos = auc_between(t_fit, y_fit, sos_t, eos_t)
        auc_above_base_full = float(np.trapz(np.maximum(0.0, y_fit - ndvi_base), t_fit) * DAYS_FOR_PLOT)
        auc_above_base_sos_eos = auc_between(t_fit, np.maximum(0.0, y_fit - ndvi_base), sos_t, eos_t)

        # --- Fit residual quality (using observed points) ---
        y_hat_obs = double_logistic_function(t, *p)
        resid = y - y_hat_obs
        rmse  = float(np.sqrt(np.nanmean((resid)**2))) if resid.size else np.nan
        iqr   = float(np.nanpercentile(resid, 75) - np.nanpercentile(resid, 25)) if resid.size else np.nan

        row = {
            "tree_id": trees[id_field].iloc[tid] if id_field else tid,
            "height": float(trees["height"].iloc[tid]),
            "area": float(trees["area"].iloc[tid]),
            # timings
            "greenup_doy": round(gu_doy, 3),
            "sos_doy": round(sos_doy, 3),
            "peak_doy": round(peak_doy, 3),
            "sen_onset_doy": round(onset_doy, 3),
            "eos_doy": round(eos_doy, 3),
            "dormancy_doy": round(dorm_doy, 3),
            "los_days": round(los_days, 3),
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
            "senescence_rate": round(senescence_rate, 9),
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
            # shade profile info
            "n_shaded_points": shade_count,
            "max_shade_points_allowed": MAX_SHADE_POINTS,
        }


        # stressors: generic pollution rasters (one column per file)
        for short_name, vals in pollution_by_layer.items():
            colname = f"poll_{short_name}"
            row[colname] = vals[tid] if tid < len(vals) else np.nan


        # attach impervious per radius
        for r in IMPERVIOUS_BUFFERS_M:
            vals = imperv_by_r[r]
            row[imperv_cols[r]] = vals[tid] if tid < len(vals) else np.nan

        # attach temperature per radius (°C)
        for r in TEMP_BUFFERS_M:
            vals = temp_by_r[r]
            row[temp_cols[r]] = vals[tid] if tid < len(vals) else np.nan


        # attach LST temperature per radius (°C)
        for r in LST_TEMP_BUFFERS_M:
            vals = LST_temp_by_r[r]
            row[LST_temp_cols[r]] = vals[tid] if tid < len(vals) else np.nan

        # attach insolation per radius (°C)
        for r in INS_BUFFERS_M:
            vals = insolation_by_r[r]
            row[insolation_cols[r]] = vals[tid] if tid < len(vals) else np.nan

        metrics_rows.append(row)

        # Plot NDVI + 3rd derivative for debugging SOS
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 6.0), sharex=True)

        # --- Top panel: NDVI + phenology markers ---
        ax1.scatter(x_doy, y, s=22, alpha=0.9, label="Observed NDVI")
        ax1.plot(t_fit * DAYS_FOR_PLOT, y_fit, linewidth=2, label="Robust fit")

        #ax1.axvline(sos_doy,   linestyle="--", linewidth=1.5, label=f"SOS ≈ {sos_doy:.1f}")
        #ax1.axvline(peak_doy,  linestyle="--", linewidth=1.5, label=f"Peak ≈ {peak_doy:.1f}")
        #ax1.axvline(eos_doy,   linestyle="--", linewidth=1.5, label=f"EOS ≈ {eos_doy:.1f}")
        #ax1.axvline(onset_doy, linestyle="--", linewidth=1.5, label=f"SO ≈ {onset_doy:.1f}")

        ax1.set_ylabel("NDVI")
        ax1.set_title(
            f"Tree {row['tree_id']} — SOS/POS/EOS + metrics (n_shaded={shade_count})"
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # --- Bottom panel: 3rd derivative (y3) + events ---
        x_fit_doy = t_fit * DAYS_FOR_PLOT
        ax2.plot(x_fit_doy, y3, linewidth=1.5, label="3rd derivative (y3)")

        # Mark events
        ax2.axvline(gu_doy,   color="gray", linestyle=":",  linewidth=1.0, label="Green-up")
        ax2.axvline(sos_doy,  color="C0",   linestyle="--", linewidth=1.2, label="SOS (1st pit)")
        ax2.axvline(peak_doy, color="C1",   linestyle="--", linewidth=1.2, label="Maturity (2nd peak)")
        ax2.axvline(onset_doy,color="C2",   linestyle="--", linewidth=1.2, label="Senescence (2nd pit)")
        ax2.axvline(eos_doy,  color="C3",   linestyle="--", linewidth=1.2, label="EOS (3rd peak)")
        ax2.axvline(dorm_doy, color="C4",   linestyle=":",  linewidth=1.0, label="Dormancy (3rd pit)")

        ax2.set_xlabel("Day of Year")
        ax2.set_ylabel("y3")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7)

        out_png = os.path.join(plot_dir, f"tree_{row['tree_id']}_ndvi_fit.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"[Saved] {out_png}")

    # Save metrics CSV
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(METRICS_CSV_NAME, index=False)
    print(f"[OK] Saved metrics CSV with stressor columns "
          f"{list(imperv_cols.values()) + list(temp_cols.values())}: {METRICS_CSV_NAME}")

    # ---------------------------------------------------------
    # Join phenology + environmental metrics back to tree layer
    # ---------------------------------------------------------
    if "tree_id" not in df_metrics.columns:
        raise RuntimeError("df_metrics does not contain 'tree_id' column; cannot join to trees.")

    # Merge on the same ID field you used when building metrics_rows
    # (tree_id = trees[id_field] for each tid)
    if id_field is not None:
        trees_with_metrics = trees.merge(
            df_metrics,
            how="left",
            left_on=id_field,
            right_on="tree_id"
        )
    else:
        # Fallback: join on row index if no explicit ID field exists
        trees_with_metrics = trees.join(
            df_metrics.set_index("tree_id"),
            how="left"
        )

    # Save updated tree layer with all metrics + stressors as attributes
    trees_with_metrics.to_file(OUTPUT_TREE_LAYER_PATH, driver="GPKG")
    print(f"[OK] Saved tree layer with phenology metrics and environmental variables: {OUTPUT_TREE_LAYER_PATH}")
    print("[Done]")



if __name__ == "__main__":
    main()
