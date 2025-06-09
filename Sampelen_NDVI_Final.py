import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import os 
import pandas as pd
import numpy as np
from osgeo import gdal

shapefile_path = "/Users/alexsamyn/Documents/BAP_(Mac)/Clusters_Final/CL_PM25_Final_Con.geojson"
raster_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Indices/NDVI"
output_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/NDVI"


def search_highest_reflectance(ds_array, i, j, objn):
    cutout = ds_array[:,i-1:i+2,j-1:j+2]
    try:
        if len(cutout[:,1,1] > 4):
            ndvi = (cutout[7]-cutout[5])/(cutout[7]+cutout[5])
        else:
            ndvi = (cutout[3]-cutout[2])/(cutout[3]+cutout[2])
        ndvi[ndvi < 0.1] = 0
        ndvi[ndvi >= 0.1] = 1
        cutout = cutout * ndvi
    except:
        print('ndvi calculation not possible')
    cutout = cutout.sum(axis=0)
    i_c, j_c, _ = cutout.argmax(axis=0)
    i = (i-1) + i_c
    j = (j-1) + j_c
    return i, j

def get_indices(x, y, ox, oy, pw, ph):
    i = np.floor((oy - y) / ph).astype('int')
    j = np.floor((x - ox) / pw).astype('int')
    return i, j

def sample_points(x, y, ds, objn, ndvi=True):
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
    x_coords = gdf['geometry'].x
    y_coords = gdf['geometry'].y
    anlag_objn = gdf['field_1']
    ds = gdal.Open(raster_path, 0)
    sampled = sample_points(x_coords, y_coords, ds, anlag_objn, ndvi)
    del ds
    return sampled

def sample_rasters(shapefile_path, raster_folder, ndvi=True):
    sampled_all = {}
    for raster in os.listdir(raster_folder):
        if not raster.lower().endswith(".tif"):
            continue  # sla .DS_Store of andere bestanden over

        full_path = os.path.join(raster_folder, raster)
        print(f"Working on raster {raster}")
        try:
            sampled = sample_raster(shapefile_path, full_path, ndvi)
            basename = os.path.splitext(raster)[0]
            try:
                date_str = basename.split('_')[2]
                date_formatted = pd.to_datetime(date_str, format='%Y%m%d').strftime('%d_%m_%Y')
            except Exception as e:
                print(f"Datumformaat onjuist in bestandsnaam: {raster}")
                print(e)
                continue

            if ndvi:
                sampled_all[date_formatted] = sampled
                print(date_formatted)
            else:
                for i in range(len(sampled[:, 1])):
                    sampled_all[f"{date_formatted}.{i}"] = sampled[i, :]
        except Exception as error:
            print(raster + " could not be opened")
            print(error)
    df = pd.DataFrame(sampled_all)
    return df

# MAIN BLOK
ndvi = True  # Zet op False voor reflectantie

# 1. Extract NDVI/reflectantie per boom
df = sample_rasters(shapefile_path, raster_folder, ndvi=ndvi)

# 2. Voeg id en ClusterID toe als kolommen
gdf = gpd.read_file(shapefile_path)
df['field_1'] = gdf['field_1'].values
df['CL_ID'] = gdf['CL_ID'].values

# 3. Zet id en ClusterID vooraan
data_columns = [col for col in df.columns if col not in ['field_1', 'CL_ID']]

# 4. Sorteer de overige kolommen (op datum of datum+band)
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

# 5. Opslaan als CSV
df.to_csv('NDVI_per_boom.csv', index=False)

output_path = os.path.join(output_folder, 'NDVI_per_boom.csv')
df.to_csv(output_path, index=False)

print(f" CSV  opgeslagen als '{output_path}'")
