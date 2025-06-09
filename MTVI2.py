
import numpy as np
import rasterio
import os
from glob import glob

# === Instellingen ===
input_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/RC"
output_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Indices/MTVI2"
os.makedirs(output_folder, exist_ok=True)

# === BATCHVERWERKING ===
raster_files = glob(os.path.join(input_folder, "*.tif"))

for raster_path in raster_files:
    filename = os.path.basename(raster_path)
    output_path = os.path.join(output_folder, filename.replace(".tif", "_MTVI2.tif"))
    print(f" Verwerk: {filename}")

    with rasterio.open(raster_path) as src:
        try:
            green = src.read(3).astype('float32')  # Band 3 = Green
            red   = src.read(6).astype('float32')  # Band 6 = Red
            nir   = src.read(8).astype('float32')  # Band 8 = NIR
            profile = src.profile
        except IndexError:
            print(f" {filename} bevat onvoldoende banden. Overgeslagen.")
            continue

        # === Reflectantieconversie
        green /= 10000
        red   /= 10000
        nir   /= 10000

        # === Clipping van extreme of ongeldige waarden
        green = np.clip(green, 0.01, 1.0)
        red   = np.clip(red, 0.01, 1.0)
        nir   = np.clip(nir, 0.01, 1.0)

        # === MTVI2-BEREKENING (stabiele formule)
        numerator = 1.5 * (1.2 * (nir - green) - 2.5 * (red - green))
        denominator = np.sqrt((2 * nir + 1)**2 - (6 * nir - 5 * red) + 1e-6)
        mtvi2 = numerator / denominator

        # === Mask foutwaarden buiten [-1, +1]
        mtvi2[(mtvi2 < -1) | (mtvi2 > 1)] = np.nan
        mtvi2[np.isnan(mtvi2)] = -9999
        mtvi2[np.isinf(mtvi2)] = -9999

        # === Opslaan als GeoTIFF
        profile.update(dtype='float32', count=1, nodata=-9999)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mtvi2, 1)

        print(f" MTVI2 opgeslagen: {output_path}")


