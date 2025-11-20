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
from pathlib import Path
from rasterio.crs import CRS

# =========================
# CONFIG
# =========================

# Path to nDSM (heights of objects, m)
NDSM_PATH = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/ndsm.tif')

# Path to tree canopy mask (1 = tree canopy, 0 = non-tree)
TREE_MASK_PATH = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/tree crown mask 3.tif')

# CSV produced by the Planet API search script
# (with columns like: composite_image_id, composite_date, sun_azimuth_deg, sun_elevation_deg, ...)
PLANET_META_CSV_PATH = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/planetscope_sun_geometry_from_api.csv')

# Output directory for shadow rasters + summary CSV
OUTPUT_DIR = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/shadow rasters')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: summary CSV of shaded fraction per composite
SUMMARY_CSV_PATH = OUTPUT_DIR / "shadow_summary_per_composite.csv"

# Name of columns in the Planet metadata CSV
COL_COMP_ID = "composite_image_id"
COL_COMP_DATE = "composite_date"
COL_AZ = "sun_azimuth_deg"
COL_EL = "sun_elevation_deg"
COL_SCENE_ID = "scene_id"
COL_CLOUD = "cloud_cover"


# =========================
# HELPER FUNCTIONS
# =========================

def load_single_band_raster(path):
    """Load a single-band raster as float32 array + profile."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
    return arr, profile


def compute_slope_aspect(ndsm, transform):
    """
    Compute slope and aspect (radians) from nDSM using a 3x3 finite difference kernel.

    Returns:
    - slope: 0 (flat) .. pi/2 (vertical)
    - aspect: 0 = north, pi/2 = east, pi = south, 3*pi/2 = west
    """
    if not isinstance(transform, Affine):
        transform = Affine(*transform)

    cellsize_x = transform.a
    cellsize_y = -transform.e  # transform.e is negative for north-up

    z = ndsm.copy()
    z[np.isinf(z)] = np.nan
    # Fill NaNs with median to avoid weird edges
    z = np.where(np.isnan(z), np.nanmedian(z), z)

    # Horn's method
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

    slope = np.full_like(z, np.nan, dtype="float32")
    aspect = np.full_like(z, np.nan, dtype="float32")

    slope_core = np.arctan(np.hypot(dzdx, dzdy))
    slope[1:-1, 1:-1] = slope_core

    # Aspect: clockwise from north
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
    - Pixel is "shaded" if cos(i) <= 0 (no direct sun) and it is a tree pixel.

    Returns:
    - shadow_on_trees: uint8 (1 = shaded tree pixel, 0 = sunlit tree or non-tree)
    """
    ndsm = ndsm.astype("float32")
    if ndsm_no_data_value is not None:
        ndsm = np.where(ndsm == ndsm_no_data_value, np.nan, ndsm)

    slope, aspect = compute_slope_aspect(ndsm, transform)

    azimuth_rad = np.deg2rad(sun_azimuth_deg)
    elevation_rad = np.deg2rad(sun_elevation_deg)
    zenith_rad = np.deg2rad(90.0 - sun_elevation_deg)

    # cos(i) = cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)
    cos_i = (
        np.cos(zenith_rad) * np.cos(slope) +
        np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    )

    # Where slope/aspect are NaN, treat as shaded (no direct illumination)
    cos_i = np.where(np.isnan(cos_i), -1.0, cos_i)

    shaded = cos_i <= 0.0

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

    # Ensure tree mask matches nDSM grid
    print("Checking tree mask grid matches nDSM grid...")
    print(f"Tree mask grid: {tree_profile['crs']}, {tree_profile['transform']}, {tree_profile['width']}x{tree_profile['height']}")
    print(f"nDSM grid: {ndsm_profile['crs']}, {ndsm_profile['transform']}, {ndsm_profile['width']}x{ndsm_profile['height']}")


    # OPTIONAL: derive tree mask from nDSM if you don't have one
    # Example: trees where height > 3 m
    # tree_mask = (ndsm > 3.0).astype("uint8")

    # Compute total tree pixels once
    tree_pixels_total = int(tree_mask.astype(bool).sum())
    if tree_pixels_total == 0:
        raise ValueError("Tree mask has zero tree pixels (no value == 1).")

    print(f"Total tree pixels in mask: {tree_pixels_total}")

    # --- Read Planet metadata CSV and average per composite ---

    if not PLANET_META_CSV_PATH.exists():
        raise FileNotFoundError(f"Planet metadata CSV not found: {PLANET_META_CSV_PATH}")

    print(f"Loading Planet metadata from {PLANET_META_CSV_PATH}...")
    meta_df = pd.read_csv(PLANET_META_CSV_PATH)

    # Basic column checks
    required_cols = [COL_COMP_ID, COL_COMP_DATE, COL_AZ, COL_EL]
    for col in required_cols:
        if col not in meta_df.columns:
            raise ValueError(f"Column '{col}' not found in {PLANET_META_CSV_PATH}")

    # Drop rows with missing sun angles
    meta_df = meta_df.dropna(subset=[COL_AZ, COL_EL])

    if meta_df.empty:
        raise ValueError("No valid rows with sun azimuth/elevation in metadata CSV.")

    # Average sun angles per composite
    # (simple arithmetic mean; fine if angles are close together in time)
    print("Averaging sun geometry per composite...")
    grouped = (
        meta_df
        .groupby([COL_COMP_ID, COL_COMP_DATE], dropna=False)
        .agg({
            COL_AZ: "mean",
            COL_EL: "mean",
            COL_SCENE_ID: "nunique",
            COL_CLOUD: "mean"
        })
        .reset_index()
        .rename(columns={
            COL_AZ: "mean_sun_azimuth_deg",
            COL_EL: "mean_sun_elevation_deg",
            COL_SCENE_ID: "n_scenes",
            COL_CLOUD: "mean_cloud_cover"
        })
    )

    if grouped.empty:
        raise ValueError("Grouped metadata is empty after aggregation.")

    print(f"Found {len(grouped)} composites with averaged sun geometry.")

    # Optionally, save the averaged angles as a helper CSV
    averaged_csv_path = OUTPUT_DIR / "planetscope_sun_geometry_averaged_per_composite.csv"
    grouped.to_csv(averaged_csv_path, index=False)
    print(f"Saved averaged sun geometry per composite to: {averaged_csv_path}")

    results = []

    # --- Main loop over composites ---
    for idx, row in grouped.iterrows():
        image_id = str(row[COL_COMP_ID])
        date_val = row[COL_COMP_DATE]
        az_deg = float(row["mean_sun_azimuth_deg"])
        el_deg = float(row["mean_sun_elevation_deg"])
        n_scenes = int(row["n_scenes"])
        mean_cloud = row["mean_cloud_cover"]

        print(
            f"\nProcessing composite {image_id} (date: {date_val}, "
            f"mean azimuth={az_deg:.2f}°, mean elevation={el_deg:.2f}°, "
            f"n_scenes={n_scenes}, mean_cloud={mean_cloud:.3f})..."
        )

        shadow_on_trees = compute_shadow_mask_on_trees(
            ndsm=ndsm,
            tree_mask=tree_mask,
            transform=transform,
            sun_azimuth_deg=az_deg,
            sun_elevation_deg=el_deg,
            ndsm_no_data_value=ndsm_nodata,
        )

        # Save raster
        ndsm_profile["crs"] = CRS.from_epsg(31370)
        out_path = OUTPUT_DIR / f"shadow_trees_{image_id}.tif"
        write_geotiff(out_path, shadow_on_trees, ndsm_profile, dtype="uint8", nodata=0)
        print(f"  -> Saved shadow mask to: {out_path}")

        # Compute summary stats
        shaded_tree_pixels = int((shadow_on_trees == 1).sum())
        shaded_fraction = shaded_tree_pixels / tree_pixels_total

        print(
            f"  -> Shaded tree pixels: {shaded_tree_pixels} "
            f"({shaded_fraction:.3%} of canopy)"
        )

        results.append({
            "composite_image_id": image_id,
            "composite_date": date_val,
            "mean_sun_azimuth_deg": az_deg,
            "mean_sun_elevation_deg": el_deg,
            "n_scenes": n_scenes,
            "mean_cloud_cover": mean_cloud,
            "tree_pixels_total": tree_pixels_total,
            "shaded_tree_pixels": shaded_tree_pixels,
            "shaded_fraction": shaded_fraction,
        })

    # Save summary CSV
    if SUMMARY_CSV_PATH is not None:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
        print(f"\nWrote shaded canopy summary to: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()