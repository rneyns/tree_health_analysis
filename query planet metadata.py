#!/usr/bin/env python3
"""
Query Planet Data API for PSScene items over Brussels AOI on given dates,
and export their solar geometry and acquisition metadata to CSV.

Inputs:
- AOI GeoJSON (FeatureCollection, single polygon is enough)
- CSV with columns: image_id,date
- Planet API key via PL_API_KEY env var (or hard-coded in the script)

Output:
- CSV with columns:
  composite_image_id, composite_date, scene_id, acquired, sun_azimuth_deg,
  sun_elevation_deg, cloud_cover, item_type
"""

import os
import json
from pathlib import Path

import requests
import pandas as pd

# =====================
# CONFIG
# =====================

# Path to your AOI GeoJSON (Brussels)
AOI_GEOJSON_PATH = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/Brussels_aoi.geojson')

# Path to your composite dates CSV
COMPOSITE_DATES_CSV = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/acquisition_dates.csv')

# Output CSV path
OUTPUT_CSV = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Shadow analysis/planetscope_sun_geometry_from_api.csv')

# Planet Data API URL
PLANET_DATA_URL = "https://api.planet.com/data/v1"

# Item types to search (you can tweak this list)
ITEM_TYPES = ["PSScene"]  # or ["PSScene", "PSScene4Band"] if needed

# Date range padding in days (to be safe around mosaic dates)
DATE_PADDING_DAYS = 0  # 0 = only that date; you can set 1 for +/- 1 day


# =====================
# AUTH
# =====================

def get_api_key():
    """Fetch Planet API key from environment variable."""
    api_key = "PLAK2e307e1fe5a94a4eb5003f70979b66ff"
    if not api_key:
        raise RuntimeError("Please set the PL_API_KEY environment variable with your Planet API key.")
    return api_key


# =====================
# HELPERS
# =====================

def load_aoi_geojson(path: Path):
    with open(path, "r") as f:
        geojson = json.load(f)

    # If FeatureCollection, use first feature
    if geojson.get("type") == "FeatureCollection":
        feature = geojson["features"][0]
        geometry = feature["geometry"]
    elif geojson.get("type") == "Feature":
        geometry = geojson["geometry"]
    else:
        geometry = geojson  # assume it's bare geometry

    return geometry


def build_date_filter(date_str: str, padding_days: int = 0):
    """
    Build a date range filter for Planet API.
    If padding_days > 0, we search a date window [date-padding, date+padding].
    """
    from datetime import datetime, timedelta

    date = datetime.strptime(date_str, "%d/%m/%Y").date()
    if padding_days > 0:
        start_date = (date - timedelta(days=padding_days)).isoformat()
        end_date = (date + timedelta(days=padding_days)).isoformat()
    else:
        start_date = date.isoformat()
        end_date = date.isoformat()

    # Planet API uses RFC3339; we’ll assume full-day window in UTC
    acquired_gte = f"{start_date}T00:00:00.000Z"
    acquired_lte = f"{end_date}T23:59:59.999Z"

    return {
        "type": "AndFilter",
        "config": [
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": acquired_gte,
                    "lte": acquired_lte
                }
            }
        ]
    }


def search_planet_items(api_key, aoi_geom, date_str, item_types, padding_days=0):
    """
    Call Planet Data API quick search for given AOI + date.
    Returns list of item features (GeoJSON).
    """
    date_filter = build_date_filter(date_str, padding_days=padding_days)

    geom_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": aoi_geom
    }

    combined_filter = {
        "type": "AndFilter",
        "config": [
            date_filter,
            geom_filter
        ]
    }

    search_endpoint = f"{PLANET_DATA_URL}/quick-search"
    headers = {
        "Content-Type": "application/json"
    }
    auth = (api_key, "")

    body = {
        "item_types": item_types,
        "filter": combined_filter
    }

    resp = requests.post(search_endpoint, auth=auth, headers=headers, json=body)
    resp.raise_for_status()

    data = resp.json()
    features = data.get("features", [])

    # If there is pagination (next page link), follow it (Planet sometimes does that)
    # but normally for small AOIs and single day, it should be all in one.
    items = features[:]
    next_link = None
    for link in data.get("_links", {}).get("next", []):
        if isinstance(link, str):
            next_link = link
        elif isinstance(link, dict) and "href" in link:
            next_link = link["href"]

    while next_link:
        resp_next = requests.get(next_link, auth=auth)
        resp_next.raise_for_status()
        data_next = resp_next.json()
        items.extend(data_next.get("features", []))

        next_link = None
        for link in data_next.get("_links", {}).get("next", []):
            if isinstance(link, str):
                next_link = link
            elif isinstance(link, dict) and "href" in link:
                next_link = link["href"]

    return items


def extract_item_metadata(item_feature):
    """
    Extract key metadata fields from a Planet item GeoJSON feature.
    """
    props = item_feature.get("properties", {})
    item_type = item_feature.get("properties", {}).get("item_type") or item_feature.get("properties", {}).get("asset_type")
    # For PSScene, scene_id is feature['id']
    scene_id = item_feature.get("id")

    acquired = props.get("acquired")
    sun_azimuth = props.get("sun_azimuth")
    sun_elevation = props.get("sun_elevation")
    cloud_cover = props.get("cloud_cover")

    return {
        "item_type": item_type,
        "scene_id": scene_id,
        "acquired": acquired,
        "sun_azimuth_deg": sun_azimuth,
        "sun_elevation_deg": sun_elevation,
        "cloud_cover": cloud_cover
    }


# =====================
# MAIN
# =====================

def main():
    api_key = get_api_key()
    print("Loaded Planet API key from environment (PL_API_KEY).")

    if not AOI_GEOJSON_PATH.exists():
        raise FileNotFoundError(f"AOI file not found: {AOI_GEOJSON_PATH}")

    if not COMPOSITE_DATES_CSV.exists():
        raise FileNotFoundError(f"Composite dates CSV not found: {COMPOSITE_DATES_CSV}")

    print(f"Loading AOI from {AOI_GEOJSON_PATH}...")
    aoi_geom = load_aoi_geojson(AOI_GEOJSON_PATH)

    print(f"Loading composite dates from {COMPOSITE_DATES_CSV}...")
    df_dates = pd.read_csv(COMPOSITE_DATES_CSV, sep=";")
    print(df_dates.head())

    if "image_id" not in df_dates.columns or "date" not in df_dates.columns:
        raise ValueError("Input CSV must have columns: 'image_id' and 'date'.")

    all_records = []

    for idx, row in df_dates.iterrows():
        composite_id = str(row["image_id"])
        date_str = str(row["date"])

        print(f"\nSearching Planet items for composite_id={composite_id}, date={date_str}...")

        items = search_planet_items(
            api_key=api_key,
            aoi_geom=aoi_geom,
            date_str=date_str,
            item_types=ITEM_TYPES,
            padding_days=DATE_PADDING_DAYS
        )

        if not items:
            print(f"  -> No Planet items found for that date and AOI.")
            continue

        print(f"  -> Found {len(items)} items.")

        for item in items:
            meta = extract_item_metadata(item)
            record = {
                "composite_image_id": composite_id,
                "composite_date": date_str,
                **meta
            }
            all_records.append(record)

    if not all_records:
        print("\nNo items found for any dates. Nothing to write.")
        return

    out_df = pd.DataFrame(all_records)

    # Sort by composite_date then acquired time
    out_df = out_df.sort_values(by=["composite_date", "acquired"])

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote Planet sun geometry metadata to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
