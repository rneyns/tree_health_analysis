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

# Band indices (1-based). Adjust to your orthos if needed.
RED_BAND   = 1
GREEN_BAND = 2
BLUE_BAND  = 3
NIR_BAND   = 4

# Thresholds / logic
USE_OTSU_FOR_NDVI = False     # auto NDVI threshold per tile
NDVI_IMP_THR      = 0.20      # NDVI < this → candidate impervious (if not using Otsu)
NDVI_VEG_THR      = 0.35      # NDVI >= this → vegetation
NDWI_WATER_THR    = 0.20      # NDWI > this → water
BSI_SOIL_THR      = 0.10      # BSI > this & NDVI in (0.1..0.35) → bare soil
SOIL_NDVI_MIN     = 0.06
SOIL_NDVI_MAX     = 0.35

# Post-processing at native resolution (5 cm)
SIEVE_MIN_PIXELS  = 100       # remove tiny blobs in masks/classes (pixels)
OUT_NODATA_UINT8  = 0         # class 0 reserved for nodata/unknown
VERBOSE           = True

# Downsampled outputs (50 cm)
WRITE_DOWNSAMPLED = True
TARGET_RES_METERS = 5      # CRS should be in meters; adjust if not
# -----------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- utilities ----
def list_tifs(root):
    files = []
    for p in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
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
    p = hist / (hist.sum() + 1e-12)
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
    if width is not None and height is not None: profile.update(width=width, height=height)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)

def get_pixel_size(transform: Affine):
    return abs(transform.a), abs(transform.e)  # assumes north-up, projected CRS

# ---- block reducers for downsampling ----
def block_mean_uint8(mask_u8, fy, fx):
    H = (mask_u8.shape[0] // fy) * fy
    W = (mask_u8.shape[1] // fx) * fx
    block = mask_u8[:H, :W]
    b4 = block.reshape(H // fy, fy, W // fx, fx)
    return b4.mean(axis=(1, 3)).astype("float32")

def block_mode_u8(classes_u8, fy, fx, num_classes=5):
    """Majority class in each block; ties go to lowest code."""
    H = (classes_u8.shape[0] // fy) * fy
    W = (classes_u8.shape[1] // fx) * fx
    block = classes_u8[:H, :W]
    h_out, w_out = H // fy, W // fx
    out = np.zeros((h_out, w_out), dtype="uint8")
    # vectorized bincount over flattened blocks
    b = block.reshape(h_out, fy, w_out, fx).transpose(0, 2, 1, 3).reshape(h_out * w_out, fy * fx)
    bc = np.apply_along_axis(lambda r: np.bincount(r, minlength=num_classes), 1, b)
    out_flat = np.argmax(bc, axis=1).astype("uint8")
    out[:] = out_flat.reshape(h_out, w_out)
    return out

# ---- main per-image processing ----
def process_image(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if VERBOSE: print(f"\n-- Processing {base}")

    with rasterio.open(path) as src:
        R = src.read(RED_BAND).astype("float32")
        G = src.read(GREEN_BAND).astype("float32")
        B = src.read(BLUE_BAND).astype("float32")
        N = src.read(NIR_BAND).astype("float32")

        # handle nodata to NaN for index math
        if src.nodata is not None:
            nod = src.nodata
            for band in (R, G, B, N):
                band[band == nod] = np.nan

        # Indices
        ndvi = safe_ratio(N, R)         # (NIR-Red)/(NIR+Red)
        ndwi = safe_ratio(G, N)         # (Green-NIR)/(Green+NIR)
        bsi  = safe_ratio((R + N), (B + G))  # Bare Soil Index

        # Threshold selection
        thr_ndvi = otsu_threshold(ndvi) if USE_OTSU_FOR_NDVI else NDVI_IMP_THR
        if VERBOSE and USE_OTSU_FOR_NDVI:
            print(f"   Otsu NDVI threshold: {thr_ndvi:.3f}")

        # Rule-based masks
        veg_mask   = ndvi >= NDVI_VEG_THR
        water_mask = ndwi > NDWI_WATER_THR
        soil_mask  = (bsi > BSI_SOIL_THR) & (ndvi > SOIL_NDVI_MIN) & (ndvi < SOIL_NDVI_MAX)
        imp_cand   = ndvi < thr_ndvi

        # Impervious (binary) at native res
        imperv = imp_cand & (~veg_mask) & (~water_mask) & (~soil_mask) & np.isfinite(ndvi) & np.isfinite(ndwi)
        imperv_u8 = imperv.astype("uint8")

        # Multiclass at native res
        # priority: water > vegetation > soil > impervious; unknown=0
        classes = np.zeros_like(imperv_u8, dtype="uint8")
        classes[water_mask] = 4
        classes[veg_mask & (~water_mask)] = 2
        classes[soil_mask & (~water_mask) & (~veg_mask)] = 3
        classes[(imperv) & (classes == 0)] = 1  # only where nothing else applied

        # Sieve small islands in the *impervious* mask and in the *class* map
        if SIEVE_MIN_PIXELS > 0:
            imperv_u8 = sieve(imperv_u8, size=SIEVE_MIN_PIXELS, connectivity=8)
            classes = sieve(classes,     size=SIEVE_MIN_PIXELS, connectivity=8)

        # Write QA indices (native)
        #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_ndvi.tif"), ndvi, src.profile, dtype="float32", nodata=np.nan)
        #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_ndwi.tif"), ndwi, src.profile, dtype="float32", nodata=np.nan)
        #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_bsi.tif"),  bsi,  src.profile, dtype="float32", nodata=np.nan)

        # Write native-res masks
        #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_impervious_5cm.tif"),
        #                  imperv_u8, src.profile, dtype="uint8", nodata=OUT_NODATA_UINT8)
        #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_classes_5cm.tif"),
        #                  classes, src.profile, dtype="uint8", nodata=OUT_NODATA_UINT8)

        if VERBOSE:
            total = imperv_u8.size
            pct_imp = 100.0 * imperv_u8.sum() / total
            print(f"   Impervious coverage (native): {pct_imp:.2f}%")

        # 50 cm downsampled outputs
        if WRITE_DOWNSAMPLED:
            px, py = get_pixel_size(src.transform)
            fx = TARGET_RES_METERS / px
            fy = TARGET_RES_METERS / py
            fx_i, fy_i = int(round(fx)), int(round(fy))

            # build 50 cm transform/shape
            new_transform = Affine(src.transform.a * fx_i, src.transform.b, src.transform.c,
                                   src.transform.d, src.transform.e * fy_i, src.transform.f)
            new_width  = int(np.floor(src.width  / fx_i))
            new_height = int(np.floor(src.height / fy_i))

            # Impervious fraction & majority via block reducers (fast if integer scale)
            if np.allclose([fx, fy], [fx_i, fy_i], atol=1e-3) and fx_i >= 1 and fy_i >= 1:
                frac = block_mean_uint8(imperv_u8, fy_i, fx_i)             # 0..1
                maj_imp = (frac >= 0.5).astype("uint8")
                maj_cls = block_mode_u8(classes, fy_i, fx_i, num_classes=5)

                write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_impervious_50cm_frac.tif"),
                                  frac, src.profile, dtype="float32", nodata=np.nan,
                                  transform=new_transform, width=frac.shape[1], height=frac.shape[0])
                #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_impervious_50cm_majority.tif"),
                #                  maj_imp, src.profile, dtype="uint8", nodata=OUT_NODATA_UINT8,
                #                  transform=new_transform, width=maj_imp.shape[1], height=maj_imp.shape[0])
                #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_classes_50cm_majority.tif"),
                #                  maj_cls, src.profile, dtype="uint8", nodata=OUT_NODATA_UINT8,
                #                  transform=new_transform, width=maj_cls.shape[1], height=maj_cls.shape[0])
                if VERBOSE:
                    print(f"   50cm outputs → fraction, majority (imp), majority (classes)")
            else:
                # Fallback with reproject (average for fraction; nearest for classes)
                frac = np.zeros((new_height, new_width), dtype="float32")
                rasterio.warp.reproject(
                    source=imperv_u8.astype("float32"), destination=frac,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=new_transform, dst_crs=src.crs,
                    resampling=Resampling.average)
                maj_imp = (frac >= 0.5).astype("uint8")

                maj_cls = np.zeros((new_height, new_width), dtype="uint8")
                rasterio.warp.reproject(
                    source=classes, destination=maj_cls,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=new_transform, dst_crs=src.crs,
                    resampling=Resampling.nearest)

                write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_impervious_50cm_frac.tif"),
                                  frac, src.profile, dtype="float32", nodata=np.nan,
                                  transform=new_transform, width=frac.shape[1], height=frac.shape[0])
                #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_impervious_50cm_majority.tif"),
                #                  maj_imp, src.profile, dtype="uint8", nodata=OUT_NODATA_UINT8,
                #                  transform=new_transform, width=maj_imp.shape[1], height=maj_imp.shape[0])
                #write_single_band(os.path.join(OUTPUT_DIR, f"{stem}_classes_50cm_majority.tif"),
                #                  maj_cls, src.profile, dtype="uint8", nodata=OUT_NODATA_UINT8,
                #                  transform=new_transform, width=maj_cls.shape[1], height=maj_cls.shape[0])

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
