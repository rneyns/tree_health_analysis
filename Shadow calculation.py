#!/usr/bin/env python3
"""
Compute cast shadows from nDSM for each Planet composite using GRASS r.sunmask.

For each composite in the Planet metadata CSV:
- Average sun azimuth / elevation per composite_id
- Run GRASS r.sunmask with those angles
- Export a GeoTIFF:
    shadow_trees_<composite_image_id>.tif

These outputs can then be used in your NDVI phenology script to sample
per-tree shade profiles.
"""

import subprocess
from pathlib import Path
import re

import pandas as pd

# ===================== CONFIG =====================

# Path to GRASS executable (from Homebrew or your symlink)
GRASS_BIN = "/opt/local/bin/grass"   # change to "grass8" etc. if needed

# GRASS database/location/mapset you created earlier
GISDBASE = Path("/Users/robbe_neyns/grassdata")
MAPSET   = "PERMANENT"

# Elevation raster name INSIDE GRASS (the nDSM you imported with r.in.gdal)
ELEV_MAP = "ndsm_31370"

# Planet metadata CSV (original file with per-scene sun angles)
PLANET_META_CSV_PATH = Path(
    "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/planetscope_sun_geometry_from_api.csv"
)

# Column names in the Planet metadata CSV
COL_COMP_ID   = "composite_image_id"
COL_COMP_DATE = "composite_date"
COL_AZ        = "sun_azimuth_deg"
COL_EL        = "sun_elevation_deg"
COL_SCENE_ID  = "scene_id"
COL_CLOUD     = "cloud_cover"

# Output directory for cast-shadow rasters
OUTPUT_DIR = Path(
    "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/shadow rasters"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: save averaged per-composite sun geometry
OUT_AVG_CSV = OUTPUT_DIR / "planetscope_sun_geometry_averaged_per_composite.csv"

# ==================================================


def run_grass_module(args):
    """
    Run a GRASS module via 'grass --exec'.

    args: list of strings, e.g.
        ["r.sunmask", "elevation=...", "output=...", "azimuth=...", "altitude=..."]
    """
    locpath = GISDBASE / MAPSET
    cmd = [GRASS_BIN, str(locpath), "--exec"] + args
    print(">>", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def sanitize_grass_name(name):
    """
    Make a safe GRASS raster name: letters, digits, and underscores only.
    """
    # GRASS likes names starting with a letter; prefix if needed
    name = re.sub(r"[^\w]", "_", str(name))
    if not re.match(r"^[A-Za-z]", name):
        name = "r_" + name
    return name


def main():
    # ---- 1) Load metadata ----
    if not PLANET_META_CSV_PATH.exists():
        raise FileNotFoundError(f"Planet metadata CSV not found: {PLANET_META_CSV_PATH}")

    print(f"[INFO] Reading Planet metadata: {PLANET_META_CSV_PATH}")
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

    # ---- 2) Average sun angles per composite ----
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

    print(f"[INFO] Found {len(grouped)} composites with averaged sun geometry.")

    if OUT_AVG_CSV is not None:
        grouped.to_csv(OUT_AVG_CSV, index=False)
        print(f"[INFO] Saved averaged sun geometry per composite to: {OUT_AVG_CSV}")

    # ---- 3) Ensure GRASS region matches the nDSM ----
    print("[INFO] Setting GRASS region to elevation raster...")
    run_grass_module(["g.region", f"raster={ELEV_MAP}", "-p"])

    # ---- 4) Loop over composites and run r.sunmask + export ----
    for idx, row in grouped.iterrows():
        image_id = str(row[COL_COMP_ID])
        date_val = row[COL_COMP_DATE]
        az_deg   = float(row["mean_sun_azimuth_deg"])
        el_deg   = float(row["mean_sun_elevation_deg"])
        n_scenes = int(row["n_scenes"])
        mean_cloud = row["mean_cloud_cover"]

        print(
            f"\n[COMPOSITE] {image_id} "
            f"(date={date_val}, az={az_deg:.2f}°, el={el_deg:.2f}°, "
            f"n_scenes={n_scenes}, mean_cloud={mean_cloud:.3f})"
        )

        # Internal GRASS raster name (must be safe)
        grass_shadow_name = sanitize_grass_name(f"shadow_{image_id}")

        # 4a) Compute cast-shadow map with r.sunmask
        #     altitude = sun elevation (degrees above horizon)
        run_grass_module([
            "r.sunmask",
            f"elevation={ELEV_MAP}",
            f"output={grass_shadow_name}",
            f"azimuth={az_deg}",
            f"altitude={el_deg}",
            "--overwrite",
        ])

        # 4b) Export to GeoTIFF with the desired file name
        out_tif = OUTPUT_DIR / f"shadow_trees_{image_id}.tif"
        run_grass_module([
            "r.out.gdal",
            f"input={grass_shadow_name}",
            f"output={out_tif}",
            "format=GTiff",
            "createopt=COMPRESS=LZW",
            "--overwrite",
        ])

        print(f"[OK] Wrote cast-shadow raster to: {out_tif}")

    print("\n[DONE] All composites processed.")


if __name__ == "__main__":
    main()
