#!/usr/bin/env python3
"""
NDVI sampling + robust double-logistic fit per tree + phenology metrics (no env vars, no shadow).

Outputs
-------
- <OUTPUT_DIR>/ndvi_samples_wide.csv
- <OUTPUT_DIR>/ndvi_wide_columns_mapping.csv
- <OUTPUT_DIR>/ndvi_metrics.csv  (tree_id, phenology timings/magnitudes/rates/integrals/fit stats)
- <OUTPUT_DIR>/plots/tree_<id>_ndvi_fit.png
- <OUTPUT_DIR>/trees_with_pheno.gpkg  (optional join of metrics back to tree layer)
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

from scipy.optimize import least_squares
from scipy.signal import find_peaks


# --------------------------- CONFIG ---------------------------
TREE_LAYER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/platanus_x_acerifolia.shp'
TREE_LAYER_NAME   = None

NDVI_DIR          = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Planet_ndvi'
NDVI_GLOB         = "*_ndvi.tif"

OUTPUT_DIR        = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope'
OUTPUT_TREE_LAYER_PATH = os.path.join(OUTPUT_DIR, "trees_with_pheno.gpkg")

WIDE_CSV_NAME     = "ndvi_samples_wide.csv"
METRICS_CSV_NAME  = "ndvi_metrics.csv"

AUTO_SCALE_10000  = True     # divide by 10000 if NDVI looks like 0..10000
MIN_OBS_TO_FIT    = 6        # min observations to attempt a fit
SAVE_DOY_MAPPING  = True

# Optional tree filters (like the other script). Set to None to disable.
MIN_HEIGHT_M      = None     # e.g., 10
MIN_AREA_M2       = None     # e.g., 36

# Robust fitting settings (like the other script)
ROBUST_LOSS       = "soft_l1"
ROBUST_F_SCALE    = 0.2
IRLS_REFINE       = True
IRLS_ITERS        = 3

# Sampling density for metrics/plots
N_FIT_SAMPLES     = 2000
DAYS_FOR_PLOT     = 365.0
# --------------------------------------------------------------


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

    return None

def doy_and_days_in_year(date_obj):
    year = date_obj.year
    doy = int(date_obj.strftime("%j"))
    days = 366 if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) else 365
    return doy, days
# ---------------------------------------------------------


# ---------------- Double logistic model ----------------
def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    s1 = 1.0 / (1.0 + np.exp(-mS * (t - S)))
    s2 = 1.0 / (1.0 + np.exp( mA * (t - A)))
    return wNDVI + (mNDVI - wNDVI) * (s1 + s2 - 1.0)

def robust_fit_curve(t, ndvi_observed, loss="soft_l1", f_scale=0.05, max_nfev=20000, min_obs=6):
    t = np.asarray(t, float); y = np.asarray(ndvi_observed, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(y) < min_obs:
        return None

    # robust-ish initialization (percentiles)
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
        res = least_squares(
            _res, p0, bounds=(lower, upper),
            loss=loss, f_scale=f_scale, max_nfev=max_nfev
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
    t = np.asarray(t, float); y = np.asarray(y, float)
    p = np.array(p_start, float)
    lower = np.array([-1.0, -1.0, 0.00, 0.00, 0.01, 0.01], float)
    upper = np.array([ 1.50,  1.50, 1.00, 1.00, 50.0, 50.0], float)

    for _ in range(iters):
        resid = y - double_logistic_function(t, *p)
        w = _tukey_biweight(resid)
        if np.all(w == 0): break

        def _res(pp):
            return (double_logistic_function(t, *pp) - y) * np.sqrt(w)

        res = least_squares(_res, p, bounds=(lower, upper), loss="linear", max_nfev=8000)
        p = res.x
        if p[3] <= p[2]:
            p[3] = min(0.99, p[2] + 0.05)

    return p
# --------------------------------------------------------


# ------------------------ Phenology utils ----------------
def auc_between(t, y, t0, t1):
    """AUC between t0 and t1 in 'NDVI·days' (t in [0,1])."""
    if t1 <= t0:
        return 0.0
    m = (t >= t0) & (t <= t1)
    if not np.any(m):
        return 0.0
    return float(np.trapz(y[m], t[m]) * DAYS_FOR_PLOT)

def find_pheno_events_from_y3(t_fit, y3):
    """
    Mimics the other script’s y3-based event picking:
      greenup    : 1st +peak in y3
      sos        : 1st -peak after greenup
      maturity   : 2nd +peak after sos
      senescence : 2nd -peak after maturity
      eos        : 3rd +peak after senescence
      dormancy   : 3rd -peak after eos
    """
    t_fit = np.asarray(t_fit, float)
    y3 = np.asarray(y3, float)
    n = len(t_fit)

    events = dict(greenup=None, sos=None, maturity=None, senescence=None, eos=None, dormancy=None)
    if n == 0 or not np.any(np.isfinite(y3)):
        for k in events: events[k] = 0
        return events

    y3_clean = np.where(np.isfinite(y3), y3, 0.0)
    max_abs = float(np.nanmax(np.abs(y3_clean))) or 1.0
    prom = max(1e-6, 0.05 * max_abs)

    peaks, _ = find_peaks(y3_clean, prominence=prom)
    pits,  _ = find_peaks(-y3_clean, prominence=prom)
    peaks = np.array(sorted(peaks))
    pits  = np.array(sorted(pits))

    def first_after(indices, after_idx):
        if indices is None or len(indices) == 0:
            return None
        if after_idx is None:
            return int(indices[0])
        for idx in indices:
            if idx > after_idx:
                return int(idx)
        return None

    # greenup
    gu = int(peaks[0]) if peaks.size else int(np.nanargmax(y3_clean))
    events["greenup"] = gu

    # sos
    sos = first_after(pits, gu) if pits.size else None
    events["sos"] = sos

    # maturity
    if sos is not None:
        mat = first_after(peaks, sos)
        if mat is None and peaks.size > 1:
            mat = int(peaks[1])
    else:
        mat = int(peaks[1]) if peaks.size > 1 else gu
    events["maturity"] = mat

    # senescence
    sen = first_after(pits, mat) if pits.size else None
    if sen is None and pits.size > 1:
        sen = int(pits[1])
    events["senescence"] = sen

    # eos
    eos = first_after(peaks, sen if sen is not None else mat)
    if eos is None:
        if peaks.size >= 3:
            eos = int(peaks[2])
        elif peaks.size > 0:
            eos = int(peaks[-1])
        else:
            eos = n - 1
    events["eos"] = eos

    # dormancy
    dorm = first_after(pits, eos if eos is not None else (n - 1))
    if dorm is None:
        if pits.size >= 3:
            dorm = int(pits[2])
        elif pits.size > 0:
            dorm = int(pits[-1])
        else:
            dorm = n - 1
    events["dormancy"] = dorm

    # enforce non-decreasing defined indices
    last = 0
    for key in ["greenup", "sos", "maturity", "senescence", "eos", "dormancy"]:
        if events[key] is None:
            events[key] = last
        else:
            events[key] = int(np.clip(events[key], last, n - 1))
        last = events[key]

    return events
# ---------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    wide_csv_path    = os.path.join(OUTPUT_DIR, WIDE_CSV_NAME)
    mapping_csv_path = os.path.join(OUTPUT_DIR, "ndvi_wide_columns_mapping.csv")
    metrics_csv_path = os.path.join(OUTPUT_DIR, METRICS_CSV_NAME)

    # 1) Load trees (ensure points; explode MultiPoint)
    trees = gpd.read_file(TREE_LAYER_PATH, layer=TREE_LAYER_NAME) if TREE_LAYER_NAME else gpd.read_file(TREE_LAYER_PATH)
    if trees.crs is None:
        raise RuntimeError("Tree layer CRS is undefined. Please define/reproject first.")
    trees = trees[trees.geometry.notnull() & trees.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    if "MultiPoint" in trees.geometry.geom_type.unique():
        trees = trees.explode(index_parts=False)
    trees = trees.reset_index(drop=True)

    print(f"[INFO] Total trees before optional filters: {len(trees)}")

    # Optional filters (same idea as the other script, but off by default)
    if MIN_HEIGHT_M is not None:
        if "height" not in trees.columns:
            raise RuntimeError("MIN_HEIGHT_M set, but attribute 'height' not found in tree layer.")
        trees = trees[trees["height"] >= float(MIN_HEIGHT_M)].copy().reset_index(drop=True)
        print(f"[INFO] Trees retained after height filter (>= {MIN_HEIGHT_M} m): {len(trees)}")

    if MIN_AREA_M2 is not None:
        if "area" not in trees.columns:
            raise RuntimeError("MIN_AREA_M2 set, but attribute 'area' not found in tree layer.")
        trees = trees[trees["area"] >= float(MIN_AREA_M2)].copy().reset_index(drop=True)
        print(f"[INFO] Trees retained after area filter (>= {MIN_AREA_M2} m²): {len(trees)}")

    # Identify the tree id field (prefer existing IDs)
    id_field = None
    for cand in ["crown_id", "tree_id", "id", "ID"]:
        if cand in trees.columns:
            id_field = cand
            break

    # 2) Collect NDVI rasters, parse dates, sort chronologically
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
    dated.sort(key=lambda x: (x[1], x[0]))

    k = len(dated)
    col_names = ["tree_id"] + [f"DOY{i+1}" for i in range(k)]

    if SAVE_DOY_MAPPING:
        mp_rows = [{"col": f"DOY{i+1}", "date": d.isoformat(), "doy": doy, "file": os.path.basename(f)}
                   for i, (f, d, doy, _days) in enumerate(dated)]
        pd.DataFrame(mp_rows).to_csv(mapping_csv_path, index=False)
        print(f"[OK] Saved DOY column mapping: {mapping_csv_path}")

    # 3) Sample NDVI for each tree across all images
    n_trees = len(trees)
    wide_vals = np.full((n_trees, k), np.nan, dtype=float)

    per_tree_t = [[] for _ in range(n_trees)]     # t in [0,1]
    per_tree_doy = [[] for _ in range(n_trees)]   # integer DOY
    per_tree_ndvi = [[] for _ in range(n_trees)]  # NDVI

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

        # auto-scale
        if AUTO_SCALE_10000:
            try:
                mx = np.nanmax(vals)
                if np.isfinite(mx) and mx > 2.0:
                    vals = [v / 10000.0 if np.isfinite(v) else v for v in vals]
            except Exception:
                pass

        t_frac = doy / float(days)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                wide_vals[i, j] = v
                per_tree_t[i].append(t_frac)
                per_tree_doy[i].append(doy)
                per_tree_ndvi[i].append(v)

    # 4) Write wide CSV
    tree_ids = trees[id_field].tolist() if id_field else list(range(n_trees))
    df_wide = pd.DataFrame(
        [[tree_ids[i]] + [None if np.isnan(wide_vals[i, j]) else float(wide_vals[i, j]) for j in range(k)]
         for i in range(n_trees)],
        columns=col_names
    )
    df_wide.to_csv(wide_csv_path, index=False)
    print(f"[OK] Saved wide CSV: {wide_csv_path}")

    # 5) Fit + metrics + plots
    metrics_rows = []

    for i in range(n_trees):
        t = np.asarray(per_tree_t[i], dtype=float)
        y = np.asarray(per_tree_ndvi[i], dtype=float)
        x_doy = np.asarray(per_tree_doy[i], dtype=float)

        m = np.isfinite(t) & np.isfinite(y) & np.isfinite(x_doy)
        t, y, x_doy = t[m], y[m], x_doy[m]

        if len(y) < MIN_OBS_TO_FIT:
            continue

        p = robust_fit_curve(t, y, loss=ROBUST_LOSS, f_scale=ROBUST_F_SCALE, min_obs=MIN_OBS_TO_FIT)
        if p is None:
            continue
        if IRLS_REFINE:
            p = irls_refine(t, y, p, iters=IRLS_ITERS)

        # Dense curve + derivatives
        t_fit = np.linspace(0.0, 1.0, N_FIT_SAMPLES)
        y_fit = double_logistic_function(t_fit, *p)
        y1 = np.gradient(y_fit, t_fit)
        y2 = np.gradient(y1, t_fit)
        y3 = np.gradient(y2, t_fit)

        wNDVI, mNDVI, S_est, A_est, mS_est, mA_est = map(float, p)
        ndvi_base = float(wNDVI)
        ndvi_max  = float(mNDVI)
        amplitude = float(ndvi_max - ndvi_base)

        # y3-based events
        ev = find_pheno_events_from_y3(t_fit, y3)
        gu_idx, sos_idx, mat_idx = ev["greenup"], ev["sos"], ev["maturity"]
        sen_idx, eos_idx, dorm_idx = ev["senescence"], ev["eos"], ev["dormancy"]

        # Convert to DOY scale
        gu_t, sos_t, peak_t = float(t_fit[gu_idx]), float(t_fit[sos_idx]), float(t_fit[mat_idx])
        onset_t, eos_t, dorm_t = float(t_fit[sen_idx]), float(t_fit[eos_idx]), float(t_fit[dorm_idx])

        gu_doy, sos_doy, peak_doy = gu_t * DAYS_FOR_PLOT, sos_t * DAYS_FOR_PLOT, peak_t * DAYS_FOR_PLOT
        onset_doy, eos_doy, dorm_doy = onset_t * DAYS_FOR_PLOT, eos_t * DAYS_FOR_PLOT, dorm_t * DAYS_FOR_PLOT

        # NDVI at those times
        ndvi_gu    = float(np.interp(gu_t,   t_fit, y_fit))
        ndvi_sos   = float(np.interp(sos_t,  t_fit, y_fit))
        ndvi_peak  = float(np.interp(peak_t, t_fit, y_fit))
        ndvi_onset = float(np.interp(onset_t,t_fit, y_fit))
        ndvi_eos   = float(np.interp(eos_t,  t_fit, y_fit))
        ndvi_dorm  = float(np.interp(dorm_t, t_fit, y_fit))

        # durations / asymmetry
        los_days      = float(max(0.0, eos_doy - sos_doy))
        greenup_days  = float(max(0.0, peak_doy - sos_doy))
        plateau_days  = float(max(0.0, onset_doy - peak_doy))
        decline_days  = float(max(0.0, eos_doy - onset_doy))
        asymmetry     = float((decline_days - greenup_days) / los_days) if los_days > 0 else np.nan

        # rates
        dt_days = max(1e-6, peak_doy - sos_doy)
        slope_sos_peak = (ndvi_peak - ndvi_sos) / dt_days

        # senescence rates
        i_range = np.arange(sen_idx, len(t_fit))
        min_slope_after_onset = float(np.nanmin(y1[i_range])) if i_range.size else float(np.nanmin(y1))
        senescence_rate = min_slope_after_onset * DAYS_FOR_PLOT
        mean_senescence_rate = (ndvi_eos - ndvi_onset) / max(1e-6, decline_days)

        # integrals
        auc_full = float(np.trapz(y_fit, t_fit) * DAYS_FOR_PLOT)
        auc_sos_eos = auc_between(t_fit, y_fit, sos_t, eos_t)
        auc_above_base_full = float(np.trapz(np.maximum(0.0, y_fit - ndvi_base), t_fit) * DAYS_FOR_PLOT)
        auc_above_base_sos_eos = auc_between(t_fit, np.maximum(0.0, y_fit - ndvi_base), sos_t, eos_t)

        # fit quality (observed points)
        y_hat_obs = double_logistic_function(t, *p)
        resid = y - y_hat_obs
        rmse  = float(np.sqrt(np.nanmean((resid)**2))) if resid.size else np.nan
        iqr   = float(np.nanpercentile(resid, 75) - np.nanpercentile(resid, 25)) if resid.size else np.nan

        row = {
            "tree_id": tree_ids[i],
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
            # fitted params (optional but handy)
            "wNDVI": round(wNDVI, 6),
            "mNDVI": round(mNDVI, 6),
            "S_est": round(S_est, 6),
            "A_est": round(A_est, 6),
            "mS_est": round(mS_est, 6),
            "mA_est": round(mA_est, 6),
            "n_obs": int(len(y)),
        }

        # If you kept height/area filters, also store them if present
        if "height" in trees.columns:
            row["height"] = float(trees["height"].iloc[i])
        if "area" in trees.columns:
            row["area"] = float(trees["area"].iloc[i])

        metrics_rows.append(row)

        # --- Plot: NDVI (top) + y3 (bottom), like the other script ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 6.0), sharex=True)

        ax1.scatter(x_doy, y, s=22, alpha=0.9, label="Observed NDVI")
        ax1.plot(t_fit * DAYS_FOR_PLOT, y_fit, linewidth=2, label="Robust fit")
        ax1.set_ylabel("NDVI")
        ax1.set_title(f"Tree {tree_ids[i]} — phenology metrics")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        x_fit_doy = t_fit * DAYS_FOR_PLOT
        ax2.plot(x_fit_doy, y3, linewidth=1.5, label="3rd derivative (y3)")
        ax2.axvline(gu_doy,   color="gray", linestyle=":",  linewidth=1.0, label="Green-up")
        ax2.axvline(sos_doy,  linestyle="--", linewidth=1.2, label="SOS")
        ax2.axvline(peak_doy, linestyle="--", linewidth=1.2, label="Maturity")
        ax2.axvline(onset_doy,linestyle="--", linewidth=1.2, label="Senescence")
        ax2.axvline(eos_doy,  linestyle="--", linewidth=1.2, label="EOS")
        ax2.axvline(dorm_doy, color="gray", linestyle=":",  linewidth=1.0, label="Dormancy")

        ax2.set_xlabel("Day of Year")
        ax2.set_ylabel("y3")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7)

        out_png = os.path.join(plot_dir, f"tree_{tree_ids[i]}_ndvi_fit.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

    # Save metrics
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"[OK] Saved metrics CSV: {metrics_csv_path}")

    # Join metrics back to tree layer (optional but matches other script structure)
    if not df_metrics.empty and "tree_id" in df_metrics.columns:
        if id_field is not None:
            trees_with_metrics = trees.merge(df_metrics, how="left", left_on=id_field, right_on="tree_id")
        else:
            trees_with_metrics = trees.join(df_metrics.set_index("tree_id"), how="left")
        trees_with_metrics.to_file(OUTPUT_TREE_LAYER_PATH, driver="GPKG")
        print(f"[OK] Saved tree layer with phenology metrics: {OUTPUT_TREE_LAYER_PATH}")

    print("[Done]")


if __name__ == "__main__":
    main()