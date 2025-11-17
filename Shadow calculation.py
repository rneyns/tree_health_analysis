#!/usr/bin/env python3
"""
Compute shaded tree-top areas for a PlanetScope time series
using an nDSM and solar geometry per date.

Outputs:
- One shade-on-trees GeoTIFF per date
- A CSV summary of shaded fraction of tree canopy per date
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from pathlib import Path

# =========================
# CONFIG
# =========================

# Path to nDSM (heights of objects, m)
NDSM_PATH = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/ndsm.tif')

# Path to tree canopy mask (1 = tree canopy, 0 = non-tree)
TREE_MASK_PATH = Path("/path/to/tree_canopy_mask.tif")

# CSV with one row per PlanetScope image and solar geometry
SUN_CSV_PATH = Path("/path/to/planetscope_sun_geometry.csv")

# Output directory for shadow masks
OUTPUT_DIR = Path("/path/to/output/shadows")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Name of columns in CSV
COL_IMAGE_ID = "image_id"
COL_DATE = "date"  # can be string; we won't parse to datetime unless you want
COL_AZIMUTH = "sun_azimuth_deg"
COL_ELEVATION = "sun_elevation_deg"

# Optional: path to write summary CSV
SUMMARY_CSV_PATH = OUTPUT_DIR / "shadow_summary_per_date.csv"


# =========================
# HELPER FUNCTIONS
# =========================

def load_single_band_raster(path):
    """Load a single-band raster as float32 array + profile."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
    return arr, profile


def ensure_same_grid(reference_profile, arr, arr_profile, resampling=Resampling.nearest):
    """
    Reproject/resample arr to match reference_profile if needed.
    Returns (arr_resampled, new_profile).
    """
    same_crs = (reference_profile["crs"] == arr_profile["crs"])
    same_transform = (reference_profile["transform"] == arr_profile["transform"])
    same_width = (reference_profile["width"] == arr_profile["width"])
    same_height = (reference_profile["height"] == arr_profile["height"])

    if same_crs and same_transform and same_width and same_height:
        return arr, arr_profile  # already aligned

    # Need to reproject to match reference
    out_arr = np.empty((reference_profile["height"], reference_profile["width"]), dtype="float32")
    out_profile = reference_profile.copy()

    with rasterio.open(arr_profile["name"]) as src:
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=out_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            resampling=resampling
        )

    return out_arr, out_profile


def compute_slope_aspect(ndsm, transform):
    """
    Compute slope and aspect (radians) from nDSM using a 3x3 finite difference kernel.
    - slope: 0 (flat) .. pi/2 (vertical)
    - aspect: 0 = north, pi/2 = east, pi = south, 3*pi/2 = west
    """
    # Cell size (assume square pixels)
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    cellsize_x = transform.a
    cellsize_y = -transform.e  # note: transform.e is negative if north-up

    # Pad edges to simplify gradients
    z = ndsm.copy()
    z[np.isinf(z)] = np.nan
    z = np.where(np.isnan(z), np.nanmedian(z), z)  # fill NaNs with median to avoid edge weirdness

    # Using Horn's method for slope/aspect
    # z1 z2 z3
    # z4 z5 z6
    # z7 z8 z9
    z1 = z[:-2, :-2]
    z2 = z[:-2, 1:-1]
    z3 = z[:-2, 2:]
    z4 = z[1:-1, :-2]
    z5 = z[1:-1, 1:-1]
    z6 = z[1:-1, 2:]
    z7 = z[2:, :-2]
    z8 = z[2:, 1:-1]
    z9 = z[2:, 2:]

    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cellsize_x)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * cellsize_y)

    # Initialize full-size arrays with NaN
    slope = np.full_like(z, np.nan, dtype="float32")
    aspect = np.full_like(z, np.nan, dtype="float32")

    # Compute slope and aspect
    # Slope
    slope_core = np.arctan(np.hypot(dzdx, dzdy))
    slope[1:-1, 1:-1] = slope_core

    # Aspect
    # Aspect measured clockwise from north: 0 = north, π/2 = east, π = south, 3π/2 = west
    aspect_core = np.arctan2(dzdy, -dzdx)
    aspect_core = np.where(aspect_core < 0, 2 * np.pi + aspect_core, aspect_core)
    aspect[1:-1, 1:-1] = aspect_core

    return slope, aspect


def compute_shadow_mask_on_trees(ndsm, tree_mask, transform,
                                 sun_azimuth_deg, sun_elevation_deg,
                                 ndsm_no_data_value=None):
    """
    Compute a binary shadow mask at tree-top level.

    Approach:
    - Compute slope and aspect from nDSM.
    - Use standard hillshade illumination model.
    - Pixel is "shaded" if cos(i) <= 0 (no direct sun) OR not tree.

    Returns:
    - shadow_on_trees: uint8 (1 = shaded tree pixel, 0 = sunlit tree or non-tree)
    """
    # Treat nodata as NaN
    ndsm = ndsm.astype("float32")
    if ndsm_no_data_value is not None:
        ndsm = np.where(ndsm == ndsm_no_data_value, np.nan, ndsm)

    # Slope + aspect
    slope, aspect = compute_slope_aspect(ndsm, transform)

    # Convert solar geometry to radians
    azimuth_rad = np.deg2rad(sun_azimuth_deg)
    elevation_rad = np.deg2rad(sun_elevation_deg)
    zenith_rad = np.deg2rad(90.0 - sun_elevation_deg)

    # Illumination cosine (standard hillshade formula)
    # cos(i) = cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)
    cos_i = (
        np.cos(zenith_rad) * np.cos(slope) +
        np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    )

    # Where slope/aspect are NaN, cos_i will be NaN -> treat as no illumination (shaded)
    cos_i = np.where(np.isnan(cos_i), -1.0, cos_i)

    # Direct shade: cos(i) <= 0
    shaded = cos_i <= 0.0

    # Restrict to trees
    tree_mask_bool = tree_mask.astype(bool)
    shadow_on_trees = np.zeros_like(tree_mask, dtype="uint8")
    shadow_on_trees[tree_mask_bool & shaded] = 1

    return shadow_on_trees


def write_geotiff(output_path, data, reference_profile, dtype="uint8", nodata=0):
    """Write 2D array to GeoTIFF using a reference profile."""
    profile = reference_profile.copy()
    profile.update(
        dtype=dtype,
        count=1,
    )
    if nodata is not None:
        profile.update(nodata=nodata)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


# =========================
# MAIN
# =========================

def main():
    print("Loading nDSM...")
    ndsm, ndsm_profile = load_single_band_raster(NDSM_PATH)
    transform = ndsm_profile["transform"]
    ndsm_nodata = ndsm_profile.get("nodata", None)

    print("Loading tree canopy mask...")
    tree_mask, tree_profile = load_single_band_raster(TREE_MASK_PATH)

    # OPTIONAL: derive tree mask from nDSM if you don't have one
    # Example: trees where height > 3 m
    # tree_mask = (ndsm > 3.0).astype("uint8")

    # Ensure tree mask matches nDSM grid (if not, you will need to reproject)
    if (
        ndsm_profile["crs"] != tree_profile["crs"]
        or ndsm_profile["transform"] != tree_profile["transform"]
        or ndsm_profile["width"] != tree_profile["width"]
        or ndsm_profile["height"] != tree_profile["height"]
    ):
        raise ValueError("Tree mask grid does not match nDSM grid. "
                         "Reproject/resample the tree mask beforehand.")

    # Load sun geometry table
    print("Loading sun geometry CSV...")
    df = pd.read_csv(SUN_CSV_PATH)

    # Basic checks
    for col in [COL_IMAGE_ID, COL_AZIMUTH, COL_ELEVATION]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {SUN_CSV_PATH}")

    # Compute total tree pixels once
    tree_pixels_total = int(tree_mask.astype(bool).sum())
    if tree_pixels_total == 0:
        raise ValueError("Tree mask has zero tree pixels (no value == 1).")

    print(f"Total tree pixels in mask: {tree_pixels_total}")

    results = []

    for idx, row in df.iterrows():
        image_id = str(row[COL_IMAGE_ID])
        az_deg = float(row[COL_AZIMUTH])
        el_deg = float(row[COL_ELEVATION])
        date_val = row.get(COL_DATE, "")

        print(f"\nProcessing image {image_id} (date: {date_val}, "
              f"azimuth={az_deg}°, elevation={el_deg}°)...")

        # Compute shadow-on-trees mask
        shadow_on_trees = compute_shadow_mask_on_trees(
            ndsm=ndsm,
            tree_mask=tree_mask,
            transform=transform,
            sun_azimuth_deg=az_deg,
            sun_elevation_deg=el_deg,
            ndsm_no_data_value=ndsm_nodata,
        )

        # Save raster
        out_path = OUTPUT_DIR / f"shadow_trees_{image_id}.tif"
        write_geotiff(out_path, shadow_on_trees, ndsm_profile, dtype="uint8", nodata=0)
        print(f"  -> Saved shadow mask to: {out_path}")

        # Compute summary stats
        shaded_tree_pixels = int((shadow_on_trees == 1).sum())
        shaded_fraction = shaded_tree_pixels / tree_pixels_total

        print(f"  -> Shaded tree pixels: {shaded_tree_pixels} "
              f"({shaded_fraction:.3%} of canopy)")

        results.append({
            "image_id": image_id,
            "date": date_val,
            "sun_azimuth_deg": az_deg,
            "sun_elevation_deg": el_deg,
            "tree_pixels_total": tree_pixels_total,
            "shaded_tree_pixels": shaded_tree_pixels,
            "shaded_fraction": shaded_fraction,
        })

    # Save summary CSV
    if SUMMARY_CSV_PATH is not None:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
        print(f"\nWrote summary table to: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
