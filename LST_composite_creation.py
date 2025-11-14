#!/usr/bin/env python3
import os, glob, re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from collections import defaultdict
from datetime import datetime

# ---------- user settings ----------
INPUT_DIR = r'/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/LST landsat'
OUTPUT_DIR = r'/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/LST landsat composite'
OUT_NODATA  = -9999.0


MAKE_SEASONAL = True
MAKE_ANNUAL   = True

DO_MEAN = True
DO_MIN  = True
DO_MAX  = True

# Masking and filtering
MIN_COUNT = 3                 # require at least N valid pixels
TRIM_PCT  = (0.05, 0.05)      # trimmed mean percentiles (0 disables)
VERBOSE   = True
# -----------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Helper functions ==========

def gather_rasters(root):
    """Recursively list all .tif/.tiff files"""
    exts = ("*LST.tif", "*.TIF", "*.tiff", "*.TIFF")
    files = []
    for e in exts:
        files.extend(glob.iglob(os.path.join(root, "**", e), recursive=True))
    return sorted(set(files))

# Seasons
SEASONS = {"DJF": (12,1,2), "MAM": (3,4,5), "JJA": (6,7,8), "SON": (9,10,11)}
def month_to_season(m):
    for s, months in SEASONS.items():
        if m in months: return s
    raise ValueError(m)

# Date parsing (YYYYMMDD_ prefix)
DATE_PREFIX_RE = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_")
def parse_date_from_name(path):
    base = os.path.basename(path)
    m = DATE_PREFIX_RE.match(base)
    if not m: return None
    y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
    try:
        datetime(y, mo, d)
    except ValueError:
        return None
    return y, mo, d

# ========= ALIGNMENT STEP ==========

import shutil

def align_rasters_to_reference(file_list, out_dir, root_dir):
    """
    Warp/copy all rasters to the grid of the first raster in file_list.
    Preserves the relative path under `root_dir` inside `out_dir` to avoid name collisions.
    Always produces one aligned file per input (copy if already aligned).
    Returns (aligned_files, ref_profile).
    """
    if not file_list:
        raise ValueError("No rasters to align.")
    os.makedirs(out_dir, exist_ok=True)

    # reference grid from first raster
    ref_path = file_list[0]
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_profile = ref.profile.copy()

    aligned_files = []

    for src_path in file_list:
        # Build a unique destination path by mirroring the tree under root_dir
        rel = os.path.relpath(src_path, root_dir)
        dest_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with rasterio.open(src_path) as src:
            same_grid = (
                (src.crs == ref_crs) and
                (src.width == ref_width) and
                (src.height == ref_height) and
                src.transform.almost_equals(ref_transform)
            )

            if same_grid:
                # Just copy the file; guarantees one output per input
                if src_path != dest_path:
                    shutil.copy2(src_path, dest_path)
                else:
                    # same path (unlikely), write a true copy to avoid in-place edits later
                    tmp_copy = dest_path + ".copy.tif"
                    shutil.copy2(src_path, tmp_copy)
                    shutil.move(tmp_copy, dest_path)
            else:
                if VERBOSE:
                    print(f"[align] {os.path.basename(src_path)} → reference grid")

                # Prepare destination profile to match reference grid
                dst_profile = ref_profile.copy()
                dst_profile.update(
                    crs=ref_crs,
                    transform=ref_transform,
                    width=ref_width,
                    height=ref_height,
                    dtype="float32",
                    nodata=(src.nodata if src.nodata is not None else 0),
                    compress="deflate",
                    predictor=2,
                    tiled=True,
                    count=1,
                )

                with rasterio.open(dest_path, "w", **dst_profile) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear,  # LST is continuous
                        src_nodata=src.nodata,
                        dst_nodata=dst_profile["nodata"],
                    )

        aligned_files.append(dest_path)

    return aligned_files, ref_profile


# ========= Compositing functions ==========

def read_masked(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        arr = np.ma.masked_equal(data, 0)
        if src.nodata is not None:
            arr = np.ma.masked_equal(arr, src.nodata)
        return arr

def stack_masked(files):
    layers = [read_masked(f) for f in files]
    stacked = np.ma.stack(layers, axis=0)
    valid_count = (~stacked.mask).sum(axis=0).astype("float32")
    return stacked, valid_count

def trimmed_mean(stacked, low_pct, high_pct):
    if (low_pct, high_pct) == (0, 0):
        return np.ma.mean(stacked, axis=0)
    data = stacked.filled(np.nan)
    low_q = low_pct * 100.0
    high_q = 100.0 - (high_pct * 100.0)
    low_v = np.nanpercentile(data, low_q, axis=0)
    high_v = np.nanpercentile(data, high_q, axis=0)
    mask_low, mask_high = data < low_v, data > high_v
    data_masked = np.ma.array(data, mask=(np.isnan(data) | mask_low | mask_high))
    return np.ma.mean(data_masked, axis=0)

def write_band(array, path, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)

def finalize_and_write(basename, stacked, valid_count, profile):
    count_mask = valid_count < float(MIN_COUNT)

    if DO_MEAN:
        mean_arr = trimmed_mean(stacked, *TRIM_PCT)
        mean_arr = np.ma.array(mean_arr, mask=count_mask)
        write_band(mean_arr.filled(OUT_NODATA), os.path.join(OUTPUT_DIR, f"{basename}_mean.tif"), profile)

    if DO_MIN:
        min_arr = np.ma.min(stacked, axis=0)
        min_arr = np.ma.array(min_arr, mask=count_mask)
        write_band(min_arr.filled(OUT_NODATA), os.path.join(OUTPUT_DIR, f"{basename}_min.tif"), profile)

    if DO_MAX:
        max_arr = np.ma.max(stacked, axis=0)
        max_arr = np.ma.array(max_arr, mask=count_mask)
        write_band(max_arr.filled(OUT_NODATA), os.path.join(OUTPUT_DIR, f"{basename}_max.tif"), profile)

    write_band(np.where(count_mask, OUT_NODATA, valid_count),
               os.path.join(OUTPUT_DIR, f"{basename}_count.tif"), profile)

# ========== MAIN ==========

all_files = gather_rasters(INPUT_DIR)
if not all_files:
    raise SystemExit(f"No rasters found in {INPUT_DIR}")

if VERBOSE:
    print(f"Found {len(all_files)} rasters, aligning to common grid...")

aligned_dir = os.path.join(OUTPUT_DIR, "aligned_temp")
aligned_files, ref_profile = align_rasters_to_reference(all_files, aligned_dir,INPUT_DIR)

# Group aligned rasters
season_groups = defaultdict(list)
annual_groups = defaultdict(list)
for f in aligned_files:
    parsed = parse_date_from_name(f)
    if not parsed:
        continue
    y, m, _ = parsed
    season_groups[month_to_season(m)].append(f)
    annual_groups[y].append(f)

ref_profile.update(dtype="float32", count=1, nodata=OUT_NODATA, compress="deflate", predictor=2, tiled=True)

# Seasonal composites
if MAKE_SEASONAL:
    for s in ["DJF","MAM","JJA","SON"]:
        files = season_groups[s]
        if not files: continue
        if VERBOSE: print(f"[{s}] {len(files)} rasters")
        stacked, vcount = stack_masked(files)
        finalize_and_write(f"LST_{s}", stacked, vcount, ref_profile)

# Annual composites
if MAKE_ANNUAL:
    for y in sorted(annual_groups.keys()):
        files = annual_groups[y]
        if not files: continue
        if VERBOSE: print(f"[{y}] {len(files)} rasters")
        stacked, vcount = stack_masked(files)
        finalize_and_write(f"LST_{y}", stacked, vcount, ref_profile)

print("✅ Done. Aligned rasters and seasonal/annual composites written to:")
print(OUTPUT_DIR)
