#!/usr/bin/env python3
"""
Compute NDVI and MTVI2 for PlanetScope SuperDove (8-band) images in a folder.
- Scans INPUT_FOLDER for GeoTIFFs (optionally recursive)
- Writes *_ndvi.tif and *_mtvi2.tif to OUTPUT_FOLDER
- Preserves CRS/transform; outputs are single-band float32 with LZW

Requires: rasterio, numpy, tqdm
pip install rasterio numpy tqdm
"""

import os
from pathlib import Path
import math
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ----------------------------
# CONFIG — EDIT THESE
# ----------------------------
INPUT_FOLDER = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Alex PlanetScope corrected')
OUTPUT_FOLDER = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Planet_ndvi')

# PlanetScope SuperDove (8B) default band indices (1-based in rasterio)
BAND_GREEN = 4  # Green
BAND_RED = 6    # Red
BAND_NIR = 8    # NIR

# Search options
RECURSIVE = False
GLOB = "*.tif"  # or "*.tiff"

# Output suffixes
SUFFIX_NDVI = "ndvi"
SUFFIX_MTVI2 = "mtvi2"
# ----------------------------


def _auto_to_reflectance(arr: np.ndarray) -> np.ndarray:
    """Auto-normalize scaled integers (e.g., uint16 scaled by 10000) to [0,1]."""
    arr = arr.astype("float32")
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    p95 = np.nanpercentile(arr[finite], 95)
    if p95 > 1.5:  # heuristic: looks like scaled reflectance
        arr /= 10000.0
    return arr


def _safe_div(numer, denom):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(numer, denom)
        out[~np.isfinite(out)] = np.nan
    return out


def compute_ndvi(nir, red):
    # NDVI = (NIR - Red) / (NIR + Red)
    return _safe_div(nir - red, nir + red)


def compute_mtvi2(nir, red, green):
    # MTVI2 (Haboudane 2004):
    # 1.5 * [1.2*(NIR - G) - 2.5*(R - G)] /
    # sqrt((2*NIR + 1)^2 - (6*NIR - 5*sqrt(R)) - 0.5)
    top = 1.2 * (nir - green) - 2.5 * (red - green)
    term = (2.0 * nir + 1.0) ** 2 - (6.0 * nir - 5.0 * np.sqrt(np.clip(red, 0, None))) - 0.5
    term = np.clip(term, 1e-12, None)  # avoid negatives/zero
    return 1.5 * _safe_div(top, np.sqrt(term))


def read_bands(src, bands):
    """Read specified 1-based band indices to float32 reflectance arrays."""
    arrs = []
    for b in bands:
        if b < 1 or b > src.count:
            raise ValueError(f"Requested band {b} not present (image has {src.count} bands).")
        a = src.read(b).astype("float32")
        a = _auto_to_reflectance(a)
        arrs.append(a)
    return arrs


def write_singleband_like(src, out_path, data):
    profile = src.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        compress="lzw",
        predictor=3,   # better for float
    )
    # Let NaNs represent NoData; many tools respect this without explicit nodata.
    profile["nodata"] = None

    data = data.astype("float32")
    data[~np.isfinite(data)] = np.nan

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


def is_raster(path: Path):
    return path.suffix.lower() in {".tif", ".tiff"}


def process_image(in_path: Path, out_dir: Path,
                  band_green=BAND_GREEN, band_red=BAND_RED, band_nir=BAND_NIR):
    with rasterio.open(in_path) as src:
        if src.count < max(band_green, band_red, band_nir):
            raise ValueError(
                f"{in_path.name}: expected ≥ {max(band_green, band_red, band_nir)} bands, found {src.count}."
            )

        green, red, nir = read_bands(src, [band_green, band_red, band_nir])

        # Basic cleaning: set non-positive or non-finite to NaN
        for arr in (green, red, nir):
            bad = (arr <= 0) | (~np.isfinite(arr))
            arr[bad] = np.nan

        ndvi = compute_ndvi(nir, red)
        mtvi2 = compute_mtvi2(nir, red, green)

        base = in_path.stem
        ndvi_path = out_dir / f"{base}_{SUFFIX_NDVI}.tif"
        mtvi_path = out_dir / f"{base}_{SUFFIX_MTVI2}.tif"

        write_singleband_like(src, ndvi_path, ndvi)
        write_singleband_like(src, mtvi_path, mtvi2)


def main():
    in_dir = INPUT_FOLDER
    out_dir = OUTPUT_FOLDER

    if not in_dir.exists():
        raise SystemExit(f"Input folder not found: {in_dir}")

    files = list(in_dir.rglob(GLOB) if RECURSIVE else in_dir.glob(GLOB))
    files = [f for f in files if is_raster(f)]

    if not files:
        raise SystemExit("No matching GeoTIFFs found with your settings.")

    print(f"Found {len(files)} image(s). Writing outputs to: {out_dir}")

    for f in tqdm(files, desc="Processing"):
        try:
            process_image(f, out_dir)
        except Exception as e:
            print(f"⚠️  Skipped {f.name}: {e}")


if __name__ == "__main__":
    main()
