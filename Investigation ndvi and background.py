"""
Workflow (version using pre-created masked impervious raster):

1. Load ndvi_metrics_clean (per-tree table with ndvi_base, ndvi_peak, pollution, etc.).
2. Load tree crowns shapefile (polygons, one per tree, with crown_id).
3. Load *masked* impervious raster (crowns already excluded).
4. For each tree:
   - Take the tree centroid.
   - Create buffers (e.g. 5, 10, 20 m).
   - Compute mean imperviousness in those buffers using the crown-masked raster.
5. Join these impervious_% columns back into ndvi_metrics_clean and save.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats  # pip install rasterstats
from shapely.geometry import Point

# Optional: for regression later
import statsmodels.formula.api as smf

# -----------------------------------------------------------------------------
# 0. USER CONFIGURATION – EDIT THIS SECTION
# -----------------------------------------------------------------------------

# Paths
NDVI_METRICS_PATH = r'/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/tilia x euchlora/ndvi_metrics_clean.csv'
TREE_CROWNS_PATH  = r'/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/crown_shapes_final_CRS.shp'
# This is your ALREADY MASKED impervious raster (crowns excluded)
OUTPUT_MASKED_RASTER = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/impervious_crowns_masked.tif"

# Final table output
OUTPUT_CSV = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/ndvi_metrics_with_impervious.csv"

# Column names (EDIT to match your data)
TREE_ID_COL    = "tree_id"        # key in ndvi_metrics_clean
CROWN_ID_COL   = "crown_id"       # key in crowns shapefile
NDVI_BASE_COL  = "ndvi_base"
NDVI_PEAK_COL  = "ndvi_peak"
POLLUTION_COL  = "poll_no2_anmean"

# Buffer distances around the tree centroid (units = CRS units, usually metres)
BUFFER_DISTANCES = [5, 10, 20]

# If impervious raster uses nodata (set to None to auto-detect from raster)
RASTER_NODATA = None

# -----------------------------------------------------------------------------
# 1. LOAD NDVI METRICS + TREE CROWNS + MASKED IMPERVIOUS RASTER
# -----------------------------------------------------------------------------

print("Loading NDVI metrics table...")
ndvi_df = pd.read_csv(NDVI_METRICS_PATH)

print("Loading tree crowns shapefile...")
crowns = gpd.read_file(TREE_CROWNS_PATH)

print("Opening crown-masked impervious raster...")
imp_masked_src = rasterio.open(OUTPUT_MASKED_RASTER)

if RASTER_NODATA is None:
    RASTER_NODATA = imp_masked_src.nodata

# -----------------------------------------------------------------------------
# 2. ENSURE CRS ALIGNMENT & BUILD TREE CENTROIDS
# -----------------------------------------------------------------------------

if crowns.crs is None:
    raise ValueError("Tree crowns file has no CRS defined. Please set it before running.")

# Reproject crowns to raster CRS if needed
if crowns.crs != imp_masked_src.crs:
    print(f"Reprojecting crowns from {crowns.crs} to {imp_masked_src.crs}...")
    crowns = crowns.to_crs(imp_masked_src.crs)

print("Computing tree centroids...")
crowns["centroid"] = crowns.geometry.centroid
centroids_gdf = crowns[[CROWN_ID_COL, "centroid"]].copy()
centroids_gdf = centroids_gdf.set_geometry("centroid")
centroids_gdf = centroids_gdf.set_crs(crowns.crs)

# -----------------------------------------------------------------------------
# 3. COMPUTE % IMPERVIOUS AROUND TREE CENTROIDS (CROWNS ALREADY EXCLUDED)
# -----------------------------------------------------------------------------

print("Computing % impervious around tree centroids (crowns excluded in raster)...")

# Make sure centroids GeoDataFrame uses same CRS as raster
if centroids_gdf.crs != imp_masked_src.crs:
    centroids_gdf = centroids_gdf.to_crs(imp_masked_src.crs)

# Prepare columns for results
for dist in BUFFER_DISTANCES:
    centroids_gdf[f"imperv_{dist}m"] = np.nan

cent_geom = centroids_gdf.geometry

for dist in BUFFER_DISTANCES:
    print(f"  Buffer distance: {dist} m")
    buffers = cent_geom.buffer(dist)

    # Use zonal_stats on masked impervious raster
    zs = zonal_stats(
        buffers,
        OUTPUT_MASKED_RASTER,
        stats="mean",
        nodata=RASTER_NODATA,
        geojson_out=False
    )

    means = [z["mean"] if z["mean"] is not None else np.nan for z in zs]
    imperv_vals = np.array(means, dtype=float)

    # If your raster is 0–1, convert to %; if already 0–100, comment this line out
    imperv_vals = imperv_vals * 100.0

    centroids_gdf[f"imperv_{dist}m"] = imperv_vals

imperv_cols = [f"imperv_{d}m" for d in BUFFER_DISTANCES]

# -----------------------------------------------------------------------------
# 4. JOIN IMPERVIOUS COLUMNS BACK TO NDVI_METRICS_CLEAN
# -----------------------------------------------------------------------------

imperv_df = centroids_gdf[[CROWN_ID_COL] + imperv_cols].copy()

# Rename crown_id so it matches ndvi table id
imperv_df = imperv_df.rename(columns={CROWN_ID_COL: TREE_ID_COL})

print("Merging impervious metrics into NDVI metrics table...")
ndvi_merged = ndvi_df.merge(imperv_df, on=TREE_ID_COL, how="left")

# -----------------------------------------------------------------------------
# 5. OPTIONAL: EXAMPLE REGRESSIONS (NDVI_BASE / NDVI_PEAK VS POLLUTION + IMPERVIOUS)
# -----------------------------------------------------------------------------

if all(col in ndvi_merged.columns for col in [POLLUTION_COL, NDVI_BASE_COL, NDVI_PEAK_COL]):
    print("\nRunning example regressions...")

    analysis_cols = [POLLUTION_COL, NDVI_BASE_COL, NDVI_PEAK_COL] + imperv_cols
    analysis_df = (
        ndvi_merged[analysis_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    print(f"Regression dataset size: {len(analysis_df)} rows")

    # Model 1: NDVI_peak ~ pollution
    m1 = smf.ols(
        formula=f"{NDVI_PEAK_COL} ~ {POLLUTION_COL}",
        data=analysis_df
    ).fit()
    print("\nModel 1: NDVI_peak ~ pollution")
    print(m1.summary())

    # Model 2: NDVI_peak ~ pollution + imperv_10m (or first buffer)
    if "imperv_10m" in imperv_cols:
        imp_term = "imperv_10m"
    else:
        imp_term = imperv_cols[0]

    m2 = smf.ols(
        formula=f"{NDVI_PEAK_COL} ~ {POLLUTION_COL} + {imp_term}",
        data=analysis_df
    ).fit()
    print(f"\nModel 2: {NDVI_PEAK_COL} ~ {POLLUTION_COL} + {imp_term}")
    print(m2.summary())

# -----------------------------------------------------------------------------
# 6. SAVE FINAL TABLE
# -----------------------------------------------------------------------------

print(f"\nSaving merged table with impervious columns to: {OUTPUT_CSV}")
ndvi_merged.to_csv(OUTPUT_CSV, index=False)

print("Done.")
