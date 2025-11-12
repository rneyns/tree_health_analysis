#!/usr/bin/env python3
import os, glob
import numpy as np
import rasterio
from rasterio.features import sieve
from rasterio.enums import Resampling
from affine import Affine

# ------------------- USER SETTINGS -------------------
INPUT_DIR  = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/ortho data/merged_tiles"
OUTPUT_DIR = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/impervious_out"

# Band indices (1-based). Adjust if needed.
RED_BAND   = 1
GREEN_BAND = 2
BLUE_BAND  = 3
NIR_BAND   = 4

# Classification logic
USE_OTSU_FOR_NDVI = False
NDVI_IMP_THR      = 0.20    # NDVI < this → candidate impervious
NDVI_VEG_THR      = 0.35    # NDVI >= this → force vegetation
NDWI_WATER_THR    = 0.20    # NDWI > this → water

# Post-processing
SIEVE_MIN_PIXELS  = 100     # remove connected components smaller than this (#pixels)
OUT_NODATA        = 0
VERBOSE           = True

# 50 cm outputs
WRITE_DOWNSAMPLED = True
TARGET_RES_METERS = 0.50    # 0.5 m (50 cm)
# -----------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_tifs(root):
    pats = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    files = []
    for p in pats:
        files.extend(glob.iglob(os.path.join(root, "**", p), recursive=True))
    return sorted(set(files))

def safe_ratio(a, b):
    denom = a + b
    out = np.full_like(denom, np.nan, dtype="float32")
    good = ~np.isclose(denom, 0.0)
    out[good] = (a[good] - b[good]) / denom[good]
    return out

def otsu_threshold(values):
    x = values[np.isfinite(values)]
    if x.size < 1024:
        return float(np.nanmean(values))
    x = np.clip(x, -0.2, 0.9)
    hist, edges = np.histogram(x, bins=256, range=(-0.2, 0.9))
    hist = hist.astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    centers = edges[:-1] + (edges[1:] - edges[:-1]) / 2
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    return float(centers[k])

def write_single_band(path, arr, ref_profile, dtype="float32", nodata=None, transform=None, width=None, height=None):
    profile = ref_profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate", predictor=2, tiled=True)
    if transform is not None: profile.update(transform=transform)
    if width is not None and height is not None:
        profile.update(width=width, height=height)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)

def get_pixel_size(transform: Affine):
    # assumes projected CRS in meters
    px = abs(transform.a)
    py = abs(transform.e)
    return px, py

def make_downsampled_outputs(imperv_u8, src):
    """Write 50 cm impervious fraction and majority."""
    px, py = get_pixel_size(src.transform)
    # target factors (try to be robust to tiny rounding in transform)
    fx = TARGET_RES_METERS / px
    fy = TARGET_RES_METERS / py
    fx_rounded = int(round(fx))
    fy_rounded = int(round(fy))

    # compute new transform/shape
    new_transform = Affine(src.transform.a * fx_rounded, src.transform.b, src.transform.c,
                           src.transform.d, src.transform.e * fy_rounded, src.transform.f)
    new_width  = int(np.floor(src.width / fx_rounded))
    new_height = int(np.floor(src.height / fy_rounded))

    # Path stems
    base = os.path.basename(src.name)
    stem, _ = os.path.splitext(base)
    frac_path = os.path.join(OUTPUT_DIR, f"{stem}_impervious_50cm_frac.tif")
    maj_path  = os.path.join(OUTPUT_DIR, f"{stem}_impervious_50cm_majority.tif")

    # If both factors are near integers and >=1, do block-reduce (fast & exact)
    if np.allclose([fx, fy], [fx_rounded, fy_rounded], atol=1e-3) and fx_rounded >= 1 and fy_rounded >= 1:
        # trim to multiple
        H = (imperv_u8.shape[0] // fy_rounded) * fy_rounded
        W = (imperv_u8.shape[1] // fx_rounded) * fx_rounded
        block = imperv_u8[:H, :W]

        # reshape and compute block mean → fraction
        block4d = block.reshape(H // fy_rounded, fy_rounded, W // fx_rounded, fx_rounded)
        frac = block4d.mean(axis=(1, 3)).astype("float32")  # 0..1

        # majority = frac >= 0.5
        majority = (frac >= 0.5).astype("uint8")

        # write
        write_single_band(frac_path, frac, src.profile, dtype="float32",
                          nodata=np.nan, transform=new_transform, width=frac.shape[1], height=frac.shape[0])
        write_single_band(maj_path, majority, src.profile, dtype="uint8",
                          nodata=OUT_NODATA, transform=new_transform, width=majority.shape[1], height=majority.shape[0])
        if VERBOSE:
            print(f"   50cm (block) → {os.path.basename(frac_path)}, {os.path.basename(maj_path)}")
    else:
        # Fallback: reproject to target resolution, using average for fraction and mode for majority
        if VERBOSE:
            print("   Non-integer scale factor; using reproject fallback.")
        # First compute fraction by averaging in blocks via reproject with average on float
        src_float = imperv_u8.astype("float32")
        frac = np.zeros((new_height, new_width), dtype="float32")
        with rasterio.Env():
            rasterio.warp.reproject(
                source=src_float,
                destination=frac,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=Resampling.average,
            )
        majority = (frac >= 0.5).astype("uint8")
        write_single_band(frac_path, frac, src.profile, dtype="float32",
                          nodata=np.nan, transform=new_transform, width=frac.shape[1], height=frac.shape[0])
        write_single_band(maj_path, majority, src.profile, dtype="uint8",
                          nodata=OUT_NODATA, transform=new_transform, width=majority.shape[1], height=majority.shape[0])

def process_image(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if VERBOSE: print(f"\n-- Processing {base}")

    with rasterio.open(path) as src:
        R = src.read(RED_BAND).astype("float32")
        G = src.read(GREEN_BAND).astype("float32")
        B = src.read(BLUE_BAND).astype("float32")
        N = src.read(NIR_BAND).astype("float32")

        if src.nodata is not None:
            nod = src.nodata
            for band in (R, G, B, N):
                band[band == nod] = np.nan

        ndvi = safe_ratio(N, R)
        ndwi = safe_ratio(G, N)

        thr = otsu_threshold(ndvi) if USE_OTSU_FOR_NDVI else NDVI_IMP_THR

        veg_mask   = ndvi >= NDVI_VEG_THR
        water_mask = ndwi > NDWI_WATER_THR
        cand_imperv = ndvi < thr

        imperv = cand_imperv & (~veg_mask) & (~water_mask)
        imperv = np.where(np.isfinite(ndvi) & np.isfinite(ndwi), imperv, False)

        imperv_u8 = imperv.astype("uint8")

        if SIEVE_MIN_PIXELS > 0:
            imperv_u8 = sieve(imperv_u8, size=SIEVE_MIN_PIXELS, connectivity=8)

        # Outputs at native res
        write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_ndvi.tif"), ndvi, src.profile, dtype="float32", nodata=np.nan)
        write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_ndwi.tif"), ndwi, src.profile, dtype="float32", nodata=np.nan)
        write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_impervious_5cm.tif"), imperv_u8, src.profile, dtype="uint8", nodata=OUT_NODATA)

        if VERBOSE:
            pct = 100.0 * imperv_u8.sum() / imperv_u8.size
            print(f"   Impervious coverage (native): {pct:.2f}%")

        # 50 cm downsampled outputs
        if WRITE_DOWNSAMPLED:
            make_downsampled_outputs(imperv_u8, src)

def main():
    files = list_tifs(INPUT_DIR)
    if VERBOSE: print(f"Found {len(files)} GeoTIFF(s) under {INPUT_DIR}")
    if not files:
        raise SystemExit("No .tif/.tiff files found.")
    for f in files:
        process_image(f)
    print("\nDone.")

if __name__ == "__main__":
    main()
