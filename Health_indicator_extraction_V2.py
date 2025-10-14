#!/usr/bin/env python3
"""
NDVI sampling + robust double-logistic fit per tree + SOS/Peak/Slope metrics,
plus PM2.5 point sample and Impervious Fraction samples (multiple buffer radii).

Outputs
-------
- <OUTPUT_DIR>/ndvi_samples_wide.csv
- <OUTPUT_DIR>/ndvi_wide_columns_mapping.csv
- <OUTPUT_DIR>/ndvi_metrics.csv  (tree_id, sos_doy, peak_doy, ndvi_peak, slope_sos_peak, pm25, impervious_r{R}...)
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

# --------------------------- CONFIG ---------------------------
TREE_LAYER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/Acer_platanoides.shp'
TREE_LAYER_NAME   = None

NDVI_DIR          = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Planet_ndvi'
NDVI_GLOB         = "*_ndvi.tif"

OUTPUT_DIR        = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope'
WIDE_CSV_NAME     = "ndvi_samples_wide.csv"
METRICS_CSV_NAME  = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics.csv"

# Single-band PM2.5 raster (point sample)
PM25_RASTER_PATH  = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/Raster_pm25_Brussel.gpkg'  # set this

# Impervious fraction raster (0..1 or 0..100). Will output 0..1.
IMPERVIOUS_RASTER_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/imperviousness/DW_built_fraction_Brussels_2024-01-01_2024-12-31.tif'  # <-- set this (e.g., Dynamic World built-% GeoTIFF)
IMPERVIOUS_BUFFERS_M   = [0, 5, 10, 20]                       # meters; 0 = point, others = circular mean

AUTO_SCALE_10000  = True
MIN_OBS_TO_FIT    = 6
SAVE_DOY_MAPPING  = True

# Robust fitting
ROBUST_LOSS       = "soft_l1"
ROBUST_F_SCALE    = 0.05
IRLS_REFINE       = True
IRLS_ITERS        = 3

# Sampling density for metrics/plots
N_FIT_SAMPLES     = 2000
DAYS_FOR_PLOT     = 365.0
# --------------------------------------------------------------


# ---------------- Double logistic model ----------------
def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    s1 = 1.0 / (1.0 + np.exp(-mS * (t - S)))
    s2 = 1.0 / (1.0 + np.exp( mA * (t - A)))
    return wNDVI + (mNDVI - wNDVI) * (s1 + s2 - 1.0)

def robust_fit_curve(t, ndvi_observed, loss=ROBUST_LOSS, f_scale=ROBUST_F_SCALE, max_nfev=20000):
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

def irls_refine(t, y, p_start, iters=IRLS_ITERS):
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

def sample_impervious_multi_buffers(raster_path, trees_gdf, buffers_m, autoscale_0_100_to_0_1=True):
    """
    Returns a dict: {radius_m: [values per tree]} for each requested radius.
    Each list contains 0..1 fractions (auto-scales 0..100 -> 0..1 if needed).
    """
    buffers_m = list(sorted(set(float(r) for r in buffers_m)))
    out = {r: [] for r in buffers_m}

    with rasterio.open(raster_path) as src:
        pts = trees_gdf.to_crs(src.crs) if str(trees_gdf.crs) != str(src.crs) else trees_gdf

        # pixel size (meters/pixel if CRS is meter-based)
        px_w = abs(src.transform.a)
        px_h = abs(src.transform.e)
        px_size = (px_w + px_h) / 2.0
        nodata = src.nodata

        # Precompute pixel radii per requested buffer
        rad_px = {r: (0 if r <= 0 else max(1, int(round(r / max(px_size, 1e-9))))) for r in buffers_m}

        for geom in pts.geometry:
            if geom.is_empty:
                for r in buffers_m: out[r].append(np.nan)
                continue
            x, y = (geom.geoms[0].x, geom.geoms[0].y) if geom.geom_type == "MultiPoint" and len(geom.geoms) else (geom.x, geom.y)
            if np.isnan(x) or np.isnan(y):
                for r in buffers_m: out[r].append(np.nan)
                continue
            col, row = src.index(x, y)

            # For r=0 (point) we can read once
            win_pt = Window(col, row, 1, 1).intersection(Window(0, 0, src.width, src.height))
            v_point = np.nan
            if win_pt.width > 0 and win_pt.height > 0:
                arr_pt = src.read(1, window=win_pt, boundless=True, fill_value=nodata)
                v_point = float(arr_pt[0, 0])
                if nodata is not None and v_point == nodata:
                    v_point = np.nan

            for r in buffers_m:
                rp = rad_px[r]
                if r <= 0 or rp == 0:
                    v = v_point
                else:
                    win = Window(col - rp, row - rp, 2*rp + 1, 2*rp + 1).intersection(Window(0, 0, src.width, src.height))
                    if win.width <= 0 or win.height <= 0:
                        v = np.nan
                    else:
                        arr = src.read(1, window=win, boundless=True, fill_value=nodata)
                        cx = (col - win.col_off); cy = (row - win.row_off)
                        v = _mean_in_circular_window(arr, cx, cy, rp, nodata)

                # Auto-scale to 0..1 if looks like 0..100 %
                if autoscale_0_100_to_0_1 and np.isfinite(v) and v > 1.5:
                    v = v / 100.0
                out[r].append(float(v) if np.isfinite(v) else np.nan)

    return out
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

    # --- PM2.5 (point) ---
    pm25_vals = sample_pm25_per_tree(PM25_RASTER_PATH, trees)

    # --- Impervious multi-buffers ---
    imperv_by_r = sample_impervious_multi_buffers(
        IMPERVIOUS_RASTER_PATH,
        trees,
        buffers_m=IMPERVIOUS_BUFFERS_M,
        autoscale_0_100_to_0_1=True
    )
    print("[OK] Impervious buffers (m):", IMPERVIOUS_BUFFERS_M)

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
            samples = list(rio_sample(src, coords))
            vals = [float(s[0]) if (s is not None and len(s) > 0) else np.nan for s in samples]

            if AUTO_SCALE_10000:
                try:
                    mx = np.nanmax(vals)
                    if np.isfinite(mx) and mx > 2.0:
                        vals = [v / 10000.0 if np.isfinite(v) else v for v in vals]
                except Exception:
                    pass

            t_frac = doy / float(days)  # [0,1]
            for i, v in enumerate(vals):
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

    # Prepare output column names for impervious buffers
    imperv_cols = {r: f"impervious_r{int(r)}" if float(r).is_integer() else f"impervious_r{r}" for r in IMPERVIOUS_BUFFERS_M}

    for tid in range(n_trees):
        t = np.asarray(per_tree_t[tid], dtype=float)
        y = np.asarray(per_tree_ndvi[tid], dtype=float)
        x_doy = np.asarray(per_tree_doy[tid], dtype=float)
        m = np.isfinite(t) & np.isfinite(y) & np.isfinite(x_doy)
        t, y, x_doy = t[m], y[m], x_doy[m]
        if len(y) < MIN_OBS_TO_FIT:
            continue

        p = robust_fit_curve(t, y)
        if p is None:
            continue
        if IRLS_REFINE:
            p = irls_refine(t, y, p, iters=IRLS_ITERS)

        t_fit = np.linspace(0.0, 1.0, N_FIT_SAMPLES)
        y_fit = double_logistic_function(t_fit, *p)
        y3 = np.gradient(np.gradient(np.gradient(y_fit, t_fit), t_fit), t_fit)

        S_est, A_est = float(p[2]), float(p[3])
        t0 = max(0.0, S_est - 0.20)
        t1 = min(1.0, (S_est + A_est) / 2.0)
        mask = (t_fit >= t0) & (t_fit <= t1)
        if not np.any(mask):
            mask = (t_fit >= 0.0) & (t_fit <= max(0.5, S_est))
        idx_sos = np.argmax(y3[mask])
        sos_t = float(t_fit[mask][idx_sos])
        sos_doy = sos_t * DAYS_FOR_PLOT

        idx_peak = int(np.argmax(y_fit))
        peak_t = float(t_fit[idx_peak])
        peak_doy = peak_t * DAYS_FOR_PLOT
        ndvi_peak = float(y_fit[idx_peak])
        ndvi_sos = float(np.interp(sos_t, t_fit, y_fit))

        dt_days = max(1e-6, peak_doy - sos_doy)
        slope_sos_peak = (ndvi_peak - ndvi_sos) / dt_days

        row = {
            "tree_id": trees[id_field].iloc[tid] if id_field else tid,
            "sos_doy": round(sos_doy, 3),
            "peak_doy": round(peak_doy, 3),
            "ndvi_peak": round(ndvi_peak, 6),
            "slope_sos_peak": round(slope_sos_peak, 9),
            "pm25": sample if (sample := pm25_vals[tid]) is not None else np.nan,
        }

        # attach impervious for each radius
        for r in IMPERVIOUS_BUFFERS_M:
            vals = imperv_by_r[r]
            row[imperv_cols[r]] = vals[tid] if tid < len(vals) else np.nan

        metrics_rows.append(row)

        # Plot
        plt.figure(figsize=(7.8, 4.8))
        plt.scatter(x_doy, y, s=22, alpha=0.9, label="Observed NDVI")
        plt.plot(t_fit * DAYS_FOR_PLOT, y_fit, color="red", linewidth=2, label="Robust fit")
        plt.axvline(sos_doy, linestyle="--", linewidth=1.5, label=f"SOS ≈ {sos_doy:.1f}")
        plt.axvline(peak_doy, linestyle="--", linewidth=1.5, label=f"Peak ≈ {peak_doy:.1f}")
        plt.xlabel("Day of Year"); plt.ylabel("NDVI")
        plt.title(f"Tree {row['tree_id']} — SOS/Peak/Slope")
        plt.grid(True, alpha=0.3); plt.legend()
        out_png = os.path.join(plot_dir, f"tree_{tid:04d}_ndvi_fit.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
        print(f"[Saved] {out_png}")

    # Save metrics CSV
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(METRICS_CSV_NAME, index=False)
    print(f"[OK] Saved metrics CSV with impervious columns {list(imperv_cols.values())}: {METRICS_CSV_NAME}")
    print("[Done]")

if __name__ == "__main__":
    main()
