import geopandas as gpd
import rasterio
from osgeo import gdal
import os 
import pandas as pd
import numpy as np

# === Instellingen ===
shapefile_path = "/Users/alexsamyn/Documents/BAP_(Mac)/Clusters_Final/CL_PM25_Final_Con.geojson"
subselectie_path = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/SUBselectie.xlsx"
raster_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Indices/MTVI2"
output_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/MTVI2"
output_path = os.path.join(output_folder, 'MTVI2_per_boom.csv')
ndvi = True  # Laat op True staan voor eendimensionale index zoals NDVI of MTVI2

# === Laad subselectie
subselectie_df = pd.read_excel(subselectie_path)

# === Functies ===
def get_indices(x, y, ox, oy, pw, ph):
    i = np.floor((oy - y) / ph).astype('int')
    j = np.floor((x - ox) / pw).astype('int')
    return i, j

def sample_points(x, y, ds, ndvi=True):
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
    i, j = get_indices(x, y, xmin, ymax, xres, -yres)
    arr = ds.ReadAsArray()
    if ndvi:
        sampled = arr[i, j]
    else:
        sampled = arr[:, i, j]
    return sampled

def sample_raster(shapefile_path, raster_path, ndvi=True):
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf[gdf['field_1'].isin(subselectie_df['field_1'])]  # Filter op subselectie
    x_coords = gdf.geometry.x
    y_coords = gdf.geometry.y
    ds = gdal.Open(raster_path, 0)
    sampled = sample_points(x_coords, y_coords, ds, ndvi)
    del ds
    return sampled

def sample_rasters(shapefile_path, raster_folder, ndvi=True):
    sampled_all = {}
    for raster in os.listdir(raster_folder):
        if not raster.lower().endswith(".tif"):
            continue  # sla niet-rasterbestanden over

        full_path = os.path.join(raster_folder, raster)
        print(f"Working on raster {raster}")
        try:
            sampled = sample_raster(shapefile_path, full_path, ndvi)
            basename = os.path.splitext(raster)[0]
            try:
                date_str = basename.split('_')[1]  # Verwacht: 'R_YYYYMMDD_GC_RC'
                date_formatted = pd.to_datetime(date_str, format='%Y%m%d').strftime('%d_%m_%Y')
            except Exception as e:
                print(f"Datumformaat onjuist in bestandsnaam: {raster}")
                print(e)
                continue

            if ndvi:
                sampled_all[date_formatted] = sampled
                print(f"Gesampled: {date_formatted}")
            else:
                for i in range(len(sampled[:, 1])):
                    sampled_all[f"{date_formatted}.{i}"] = sampled[i, :]
        except Exception as error:
            print(f"Raster {raster} kon niet verwerkt worden:")
            print(error)
    df = pd.DataFrame(sampled_all)
    return df

# === Sampling uitvoeren
df = sample_rasters(shapefile_path, raster_folder, ndvi=ndvi)

# === Voeg 'field_1' en 'CL_ID' toe
gdf = gpd.read_file(shapefile_path)
gdf = gdf[gdf['field_1'].isin(subselectie_df['field_1'])]  # zelfde filter opnieuw
df['field_1'] = gdf['field_1'].values
df['CL_ID'] = gdf['CL_ID'].values

# === Kolommen structureren
data_columns = [col for col in df.columns if col not in ['field_1', 'CL_ID']]
if ndvi:
    sorted_cols = sorted(data_columns, key=lambda x: pd.to_datetime(x, format='%d_%m_%Y'))
else:
    sorted_cols = sorted(
        data_columns,
        key=lambda x: (
            pd.to_datetime(x.split('.')[0], format='%d_%m_%Y'),
            int(x.split('.')[1])
        )
    )

df = df[['field_1', 'CL_ID'] + sorted_cols]

# === Opslaan
os.makedirs(output_folder, exist_ok=True)
df.to_csv(output_path, index=False)
print(f" CSV succesvol opgeslagen als '{output_path}'")

