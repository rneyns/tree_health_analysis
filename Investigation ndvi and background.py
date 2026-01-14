"""
Workflow:

1) Load NDVI metrics CSV (per-tree table).
2) Load crowns polygons.
3) Filter crowns to only IDs present in CSV (speedup + avoids mismatches).
4) Build centroids from those crowns.
5) Extract impervious mean in buffers using your crown-masked impervious raster.
6) Extract LST mean in buffers 50m and 100m, ignoring zeros in the mean.
7) Merge extracted metrics into NDVI table and save CSV.
8) Join attributes back to crowns and export GeoPackage + Shapefile.

Notes:
- This script assumes the CSV key TREE_ID_COL matches crowns CROWN_ID_COL.
- ID handling is robust (string-normalized with 123.0 -> "123").
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from pathlib import Path

# Optional regression later
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# 0) USER SETTINGS
# -----------------------------------------------------------------------------

# Paths
NDVI_METRICS_PATH = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/acer platanoides/ndvi_metrics_clean.csv"
TREE_CROWNS_PATH  = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/crown_shapes_final_CRS.shp"

# Already crown-masked impervious raster
IMPERVIOUS_RASTER_PATH = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/impervious_crowns_masked.tif"

# LST raster (single-band)
LST_RASTER_PATH = r'/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/LST_July_lambert.tif'

# Outputs
OUTPUT_CSV = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer platanoides/ndvi_metrics_with_impervious.csv"
OUTPUT_GPKG = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer platanoides/trees_with_ndvi_impervious.gpkg"
GPKG_LAYER  = "trees_with_attrs"
OUTPUT_SHP  = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer platanoides/trees_with_ndvi_impervious.shp"

# Columns / IDs
TREE_ID_COL  = "tree_id"    # in CSV
CROWN_ID_COL = "crown_id"   # in crowns file

# Example regression columns (only used if they exist)
NDVI_BASE_COL  = "ndvi_base"
NDVI_PEAK_COL  = "ndvi_peak"
POLLUTION_COL  = "poll_no2_anmean"

# Impervious buffers (meters; must match CRS units)
IMPV_BUFFER_DISTANCES = [10, 20, 50, 100]

# LST buffers (meters; must match CRS units)
DO_EXTRACT_LST = True
LST_BUFFERS_M = [50, 100]

# If raster nodata missing but you know it, set it; else None
IMP_NODATA_OVERRIDE = None
LST_NODATA_OVERRIDE = None

# If your impervious raster is 0..1 and you want percent, set True
IMP_CONVERT_TO_PERCENT = True

# Optional regressions
RUN_EXAMPLE_REGRESSIONS = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def id_to_str(series: pd.Series) -> pd.Series:
    """
    Robust ID normalization:
    - turns numeric-looking IDs into integer-like strings: 123.0 -> "123"
    - leaves non-numeric IDs intact
    - returns pandas "string" dtype
    """
    s = series.copy()

    # keep original as string for fallback
    out = s.astype("string")

    # try numeric coercion
    s_num = pd.to_numeric(s, errors="coerce")
    m = s_num.notna()

    # convert numeric values to Int64 then string (handles 123.0 cleanly)
    out.loc[m] = s_num.loc[m].astype("Int64").astype("string")

    return out


def make_shp_safe_columns(gdf: gpd.GeoDataFrame, maxlen: int = 10) -> gpd.GeoDataFrame:
    """
    Shapefile field names must be <=10 chars. Rename to safe, unique names.
    """
    gdf = gdf.copy()
    geom_name = gdf.geometry.name
    used = set()
    new_cols = {}

    for c in gdf.columns:
        if c == geom_name:
            continue
        base = str(c)[:maxlen] or "field"
        candidate = base
        k = 1
        while candidate.lower() in used:
            suffix = str(k)
            candidate = (base[: maxlen - len(suffix)] + suffix)[:maxlen]
            k += 1
        used.add(candidate.lower())
        new_cols[c] = candidate

    return gdf.rename(columns=new_cols)


def prepare_single_geometry(gdf: gpd.GeoDataFrame, keep_geom: str | None = None) -> gpd.GeoDataFrame:
    """
    Ensure exactly one geometry column. Extra geometry columns become WKT strings.
    """
    gdf = gdf.copy()
    if keep_geom is None:
        keep_geom = gdf.geometry.name
    gdf = gdf.set_geometry(keep_geom)

    geom_cols = [c for c in gdf.columns if isinstance(gdf[c], gpd.GeoSeries)]
    extra = [c for c in geom_cols if c != keep_geom]
    for c in extra:
        gdf[c] = gdf[c].to_wkt()

    return gdf


def zonal_mean_ignore_zero(geoms, raster_path, nodata=None):
    """
    Compute zonal mean ignoring:
    - nodata
    - zero values
    Returns list of float (NaN if empty).
    """
    zs = zonal_stats(
        geoms,
        raster_path,
        stats=None,
        nodata=nodata,
        raster_out=True,
        geojson_out=False
    )

    means = []
    for z in zs:
        arr = z.get("mini_raster_array", None)

        if arr is None:
            means.append(np.nan)
            continue

        arr = arr.astype(float)

        # mask nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        # KEY: ignore zeros
        arr[arr == 0] = np.nan

        if np.all(np.isnan(arr)):
            means.append(np.nan)
        else:
            means.append(float(np.nanmean(arr)))

    return means


# -----------------------------------------------------------------------------
# 1) LOAD DATA
# -----------------------------------------------------------------------------

print("Loading NDVI metrics table...")
ndvi_df = pd.read_csv(NDVI_METRICS_PATH)

print("Loading crowns polygons...")
crowns = gpd.read_file(TREE_CROWNS_PATH)

if crowns.crs is None:
    raise ValueError("Crowns file has no CRS. Please define CRS before running.")


# -----------------------------------------------------------------------------
# 2) NORMALIZE IDS + FILTER CROWNS TO CSV IDS
# -----------------------------------------------------------------------------

if TREE_ID_COL not in ndvi_df.columns:
    raise ValueError(f"{TREE_ID_COL} not found in CSV columns.")

if CROWN_ID_COL not in crowns.columns:
    raise ValueError(f"{CROWN_ID_COL} not found in crowns columns.")

ndvi_df[TREE_ID_COL] = id_to_str(ndvi_df[TREE_ID_COL])
crowns[CROWN_ID_COL] = id_to_str(crowns[CROWN_ID_COL])

ndvi_ids = set(ndvi_df[TREE_ID_COL].dropna().unique())

before = len(crowns)
crowns = crowns.loc[crowns[CROWN_ID_COL].isin(ndvi_ids)].copy()
after = len(crowns)

print(f"Filtered crowns: kept {after} / {before} polygons with IDs found in the CSV.")

if len(crowns) == 0:
    raise ValueError("After filtering, zero crowns remain. Check that crown_id matches tree_id.")


# -----------------------------------------------------------------------------
# 3) OPEN RASTERS + ALIGN CRS
# -----------------------------------------------------------------------------

print("Opening impervious raster...")
imp_src = rasterio.open(IMPERVIOUS_RASTER_PATH)
imp_nodata = imp_src.nodata if IMP_NODATA_OVERRIDE is None else IMP_NODATA_OVERRIDE

# Reproject crowns to impervious CRS if needed
if crowns.crs != imp_src.crs:
    print(f"Reprojecting crowns from {crowns.crs} to {imp_src.crs}...")
    crowns = crowns.to_crs(imp_src.crs)

# Build centroids
print("Computing centroids...")
crowns["centroid"] = crowns.geometry.centroid
centroids_gdf = crowns[[CROWN_ID_COL, "centroid"]].copy()
centroids_gdf = centroids_gdf.set_geometry("centroid")
centroids_gdf = centroids_gdf.set_crs(crowns.crs)


# -----------------------------------------------------------------------------
# 4) IMPERVIOUS EXTRACTION
# -----------------------------------------------------------------------------

print("Computing impervious mean in buffers...")

for d in IMPV_BUFFER_DISTANCES:
    centroids_gdf[f"imperv_{d}m"] = np.nan

for d in IMPV_BUFFER_DISTANCES:
    print(f"  Impervious buffer: {d} m")
    buffers = centroids_gdf.geometry.buffer(d)

    zs = zonal_stats(
        buffers,
        IMPERVIOUS_RASTER_PATH,
        stats=["mean"],
        nodata=imp_nodata,
        geojson_out=False
    )
    vals = [z["mean"] if z["mean"] is not None else np.nan for z in zs]
    vals = np.array(vals, dtype=float)

    if IMP_CONVERT_TO_PERCENT:
        vals = vals * 100.0

    centroids_gdf[f"imperv_{d}m"] = vals

imperv_cols = [f"imperv_{d}m" for d in IMPV_BUFFER_DISTANCES]


# -----------------------------------------------------------------------------
# 5) LST EXTRACTION (IGNORE ZEROS)
# -----------------------------------------------------------------------------

lst_cols = [f"lst_temp_r{b}" for b in LST_BUFFERS_M]

# Ensure columns exist no matter what (prevents KeyErrors downstream)
for c in lst_cols:
    centroids_gdf[c] = np.nan

if DO_EXTRACT_LST:
    print("Extracting LST mean in buffers (zeros ignored)...")

    with rasterio.open(LST_RASTER_PATH) as lst_src:
        lst_crs = lst_src.crs
        lst_nodata = lst_src.nodata if LST_NODATA_OVERRIDE is None else LST_NODATA_OVERRIDE

    if lst_crs is None:
        raise ValueError("LST raster has no CRS defined.")

    # Reproject centroids to LST CRS if needed
    if centroids_gdf.crs != lst_crs:
        print(f"Reprojecting centroids from {centroids_gdf.crs} to {lst_crs} for LST extraction...")
        centroids_for_lst = centroids_gdf.to_crs(lst_crs)
    else:
        centroids_for_lst = centroids_gdf

    for b in LST_BUFFERS_M:
        outcol = f"lst_temp_r{b}"
        print(f"  LST buffer: {b} m -> {outcol}")

        buffers = centroids_for_lst.geometry.buffer(b)
        vals = zonal_mean_ignore_zero(buffers, LST_RASTER_PATH, nodata=lst_nodata)

        # Write back into centroids_for_lst
        centroids_for_lst[outcol] = np.array(vals, dtype=float)

    # Copy results back into centroids_gdf (index aligned)
    if centroids_for_lst is not centroids_gdf:
        for c in lst_cols:
            centroids_gdf[c] = centroids_for_lst[c].values

    print("LST extraction complete.")


# -----------------------------------------------------------------------------
# 6) MERGE EXTRACTIONS INTO NDVI TABLE
# -----------------------------------------------------------------------------

# Build imperv_df from centroids
imperv_df = centroids_gdf[[CROWN_ID_COL] + imperv_cols + lst_cols].copy()

# Normalize the join key and rename to TREE_ID_COL to match CSV
imperv_df = imperv_df.rename(columns={CROWN_ID_COL: TREE_ID_COL})
imperv_df[TREE_ID_COL] = id_to_str(imperv_df[TREE_ID_COL])  # ensure same dtype

# Merge
print("Merging extracted metrics into NDVI table...")
ndvi_merged = ndvi_df.merge(imperv_df, on=TREE_ID_COL, how="left")

print("LST columns in ndvi_merged:", [c for c in ndvi_merged.columns if c.startswith("lst_")])


# -----------------------------------------------------------------------------
# 7) OPTIONAL REGRESSIONS (SAFE AGAINST MISSING COLUMNS)
# -----------------------------------------------------------------------------

if RUN_EXAMPLE_REGRESSIONS:
    print("\nRunning example regressions (if columns exist)...")

    analysis_cols = [POLLUTION_COL, NDVI_BASE_COL, NDVI_PEAK_COL] + imperv_cols + lst_cols
    present = [c for c in analysis_cols if c in ndvi_merged.columns]
    missing = [c for c in analysis_cols if c not in ndvi_merged.columns]

    if missing:
        print("  Warning: skipping missing regression columns:", missing)

    if all(c in ndvi_merged.columns for c in [POLLUTION_COL, NDVI_PEAK_COL]):
        analysis_df = (
            ndvi_merged[present]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[POLLUTION_COL, NDVI_PEAK_COL])
        )

        print(f"  Regression dataset size: {len(analysis_df)} rows")

        # Model 1
        m1 = smf.ols(f"{NDVI_PEAK_COL} ~ {POLLUTION_COL}", data=analysis_df).fit()
        print("\nModel 1: NDVI_peak ~ pollution")
        print(m1.summary())

        # Model 2: add one impervious term + optional LST term (if present)
        imp_term = imperv_cols[0] if imperv_cols else None
        rhs_terms = [POLLUTION_COL]
        if imp_term and imp_term in analysis_df.columns:
            rhs_terms.append(imp_term)
        if "lst_temp_r100" in analysis_df.columns:
            rhs_terms.append("lst_temp_r100")

        rhs = " + ".join(rhs_terms)
        m2 = smf.ols(f"{NDVI_PEAK_COL} ~ {rhs}", data=analysis_df).fit()
        print(f"\nModel 2: {NDVI_PEAK_COL} ~ {rhs}")
        print(m2.summary())
    else:
        print("  Regression skipped: required columns not present in ndvi_merged.")


# -----------------------------------------------------------------------------
# 8) SAVE CSV
# -----------------------------------------------------------------------------

print(f"\nSaving merged table to: {OUTPUT_CSV}")
Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
ndvi_merged.to_csv(OUTPUT_CSV, index=False)
print("CSV saved.")


# -----------------------------------------------------------------------------
# 9) EXPORT SPATIAL (GPKG + SHP)
# -----------------------------------------------------------------------------

print("\nJoining attributes back to crowns and exporting spatial files...")

# Join using crown_id (already filtered)
crowns_out = crowns.merge(
    ndvi_merged,
    left_on=CROWN_ID_COL,
    right_on=TREE_ID_COL,
    how="left",
    suffixes=("", "_tbl")
)

crowns_out = prepare_single_geometry(crowns_out, keep_geom=crowns.geometry.name)

# GeoPackage
print(f"Saving GeoPackage: {OUTPUT_GPKG}")
Path(OUTPUT_GPKG).parent.mkdir(parents=True, exist_ok=True)
crowns_out.to_file(OUTPUT_GPKG, layer=GPKG_LAYER, driver="GPKG")

# Shapefile (field name limits)
print(f"Saving Shapefile: {OUTPUT_SHP}")
crowns_shp = make_shp_safe_columns(crowns_out, maxlen=10)
crowns_shp.to_file(OUTPUT_SHP, driver="ESRI Shapefile")

print("Done.")
