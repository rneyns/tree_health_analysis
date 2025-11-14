#Script that reads all tree layers in a folder and calculates the average ratio between the height and the area of the crown, to have a good idea of appropriate filter values


import geopandas as gpd
import pandas as pd
import os

folder_path = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers'

results = []

for file in os.listdir(folder_path):
    if file.lower().endswith(".shp"):
        shp_path = os.path.join(folder_path, file)
        print(f"Processing {file}...")

        try:
            gdf = gpd.read_file(shp_path)

            # Check required fields
            if not {"area", "height"}.issubset(gdf.columns):
                print(f"  Skipped: missing 'area' or 'height' column")
                continue

            # Calculate crown volume
            gdf["crown_volume"] = gdf["area"] * gdf["height"]

            # Remove invalid rows
            valid = gdf[gdf["crown_volume"] > 0]

            if len(valid) == 0:
                print("  Skipped: no valid crown_volume values")
                continue

            # Ratio = height / crown_volume
            valid["height_volume_ratio"] = valid["height"] / valid["crown_volume"]

            avg_ratio = valid["height_volume_ratio"].mean()

            results.append({
                "file": file,
                "average_ratio": avg_ratio
            })
            print(f"  Average ratio: {avg_ratio}")

        except Exception as e:
            print(f"  Error reading file: {e}")

# Convert results to a DataFrame if needed
results_df = pd.DataFrame(results)
print("\nSummary:")
print(results_df)
