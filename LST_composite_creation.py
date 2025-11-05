#!/usr/bin/env python3
import os, glob, re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from collections import defaultdict
from datetime import datetime

# ---------- user settings ----------
INPUT_DIR = r'/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/LST landsat'    # change me
OUTPUT_DIR = r'/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/LST landsat composite'       # change me
OUT_NODATA = -9999.0
ALLOW_REPROJECT = False                 # True = warp rasters to the first grid
GROUP_BY_YEAR = False                    # True = outputs like 2024_DJF_mean.tif
VERBOSE = True
# -----------------------------------


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gather rasters recursively, case-insensitive extensions
def gather_rasters(root):
    exts = ("*.tif", "*.TIF", "*.tiff", "*.TIFF")
    files = []
    for e in exts:
        files.extend(glob.iglob(os.path.join(root, "**", e), recursive=True))
    return sorted(set(files))

# Seasons (meteorological)
SEASONS = {"DJF": (12,1,2), "MAM": (3,4,5), "JJA": (6,7,8), "SON": (9,10,11)}
def month_to_season(m):
    for s, months in SEASONS.items():
        if m in months: return s
    raise ValueError(m)

# Filenames start with YYYYMMDD_*
DATE_PREFIX_RE = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_")

def parse_date_from_name(path):
    base = os.path.basename(path)
    m = DATE_PREFIX_RE.match(base)
    if not m:
        if VERBOSE: print(f"[skip:no-date-prefix] {base}")
        return None
    y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
    try:
        datetime(y, mo, d)  # sanity check
    except ValueError:
        if VERBOSE: print(f"[skip:bad-date] {base}")
        return None
    return y, mo, d

# Collect groups
groups = defaultdict(list)
all_files = gather_rasters(INPUT_DIR)
if VERBOSE:
    print(f"Found {len(all_files)} candidate rasters under {INPUT_DIR}")

for f in all_files:
    parsed = parse_date_from_name(f)
    if not parsed:
        continue
    y, m, _ = parsed
    season = month_to_season(m)
    if GROUP_BY_YEAR:
        key = ((y+1) if (season=="DJF" and m==12) else y, season)
    else:
        key = season
    groups[key].append(f)

if VERBOSE:
    for k, fs in groups.items():
        print(f"[group {k}] {len(fs)} rasters")

if not any(groups.values()):
    raise SystemExit(
        "No rasters matched the leading 'YYYYMMDD_' pattern.\n"
        "Check INPUT_DIR and that filenames begin with the date."
    )

# Reference grid
ref_file = next(iter(next(iter(groups.values()))))
with rasterio.open(ref_file) as ref:
    ref_info = {
        "profile": ref.profile.copy(),
        "transform": ref.transform,
        "crs": ref.crs,
        "shape": (ref.height, ref.width),
    }

out_profile = ref_info["profile"].copy()
out_profile.update(dtype="float32", count=1, nodata=OUT_NODATA,
                   compress="deflate", predictor=2, tiled=True)

def read_masked(path, ref=None):
    with rasterio.open(path) as src:
        if ref and ALLOW_REPROJECT and (
            src.crs != ref["crs"] or src.transform != ref["transform"] or
            src.width != ref["shape"][1] or src.height != ref["shape"][0]
        ):
            data = src.read(1, out_shape=ref["shape"], resampling=Resampling.bilinear)
        else:
            data = src.read(1)
        arr = np.ma.masked_equal(data, 0)               # mask cloud=0
        if src.nodata is not None:
            arr = np.ma.masked_equal(arr, src.nodata)   # mask source nodata
        return arr

def write_season_mean(files, key):
    stack = [read_masked(f, ref_info) for f in files]
    mean_arr = np.ma.mean(np.ma.stack(stack, axis=0), axis=0)
    out = mean_arr.filled(OUT_NODATA).astype("float32")
    if GROUP_BY_YEAR:
        year, season = key
        out_name = f"LST_{year}_{season}_mean.tif"
    else:
        season = key
        out_name = f"LST_{season}_mean.tif"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(out, 1)
    if VERBOSE:
        print(f"Wrote {os.path.basename(out_path)} from {len(files)} rasters → {out_path}")

for key in sorted(groups.keys()):
    if groups[key]:
        write_season_mean(groups[key], key)

