# Script voor radiometrische correctie en maskering van multispectrale satellietbeelden

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from imageio import imwrite  # alternatief voor visuele export
from imageio.v3 import imread
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling
from rasterio.warp import reproject
from imageio import imwrite
from skimage.exposure import match_histograms
import gc

#Inout & Output
input_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/DV_Geo_Correctie/R_M_GR"
mask_path = "/Users/alexsamyn/Documents/BAP_(Mac)/Data Brussel/Raster_Buildings_Null.tif"
output_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/Radio_Correctie/Output"
referentie_path = "/Users/alexsamyn/Documents/BAP_(Mac)/DV_Geo_Correctie/R_M_GR/R_20230405_GR.tif"

os.makedirs(output_folder, exist_ok=True)


# Stap 1: Resample masker naar referentiebeeld
def resample_mask_to_image(mask_path, referentie_path):
    with rasterio.open(referentie_path) as ref:
        ref_profile = ref.profile
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs

    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        mask_resampled = np.empty(ref_shape, dtype=mask_data.dtype)

        reproject(
            source=mask_data,
            destination=mask_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )

    return mask_resampled


# Stap 2: Verwerking per beeld
def process_image(path, mask, reference_img=None):
    with rasterio.open(path) as src:
        img = src.read()  # (bands, height, width)
        profile = src.profile

    img = np.transpose(img, (1, 2, 0))  # (height, width, bands)

    # Maskering toepassen (0 = gebouw, 1 = vegetatie)
    masked_img = img.copy()
    masked_img[mask == 0] = 0  # Alle niet-vegetatiepixels op zwart

    # Histogram matching met referentiebeeld (optioneel)
    if reference_img is not None:
        masked_img = match_histograms(masked_img, reference_img, channel_axis=-1)

    # Transpose terug naar (bands, height, width) voor opslag
    masked_img = np.transpose(masked_img, (2, 0, 1))

    return masked_img, profile


# Stap 3: Alles doorlopen en wegschrijven
def main():
    mask = resample_mask_to_image(mask_path, referentie_path)

    # Laad referentiebeeld voor histogram matching
    with rasterio.open(referentie_path) as ref_src:
        ref_img = ref_src.read()
        ref_img = np.transpose(ref_img, (1, 2, 0))

    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith(".tif"):
            continue

        path = os.path.join(input_folder, filename)
        print(f"Verwerken: {filename}")

        try:
            result_img, profile = process_image(path, mask, reference_img=ref_img)

            output_path = os.path.join(output_folder, filename)
            profile.update(dtype=result_img.dtype, count=result_img.shape[0])

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(result_img)

            print(f"Opgeslagen: {output_path}")

        except Exception as e:
            print(f"Fout bij verwerken {filename}: {e}")

        # Geheugen ruimen
        gc.collect()


if __name__ == "__main__":
    main()
