import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# === Padinstellingen ===
input_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/RC"
output_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Indices/NDVI"

os.makedirs(output_folder, exist_ok=True)

# === NDVI-functie ===
def calculate_ndvi(nir, red):
    nir = nir.astype('float32')
    red = red.astype('float32')
    ndvi = (nir - red) / (nir + red + 1e-6)  # +1e-6 om deling door nul te vermijden
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi

# === Verwerking ===
for filename in sorted(os.listdir(input_folder)):
    if not filename.endswith(".tif"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"NDVI_{filename}")

    with rasterio.open(input_path) as src:
        profile = src.profile
        profile.update(count=1, dtype='float32')

        red = src.read(6)  
        nir = src.read(8)  

        ndvi = calculate_ndvi(nir, red)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(ndvi, 1)

    print(f"NDVI opgeslagen: {output_path}")
