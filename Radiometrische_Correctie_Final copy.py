# Script voor radiometrische correctie en aangepaste visualisatie van PlanetScope-beelden

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import gc

# Padinstellingen
input_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/DV_Geo_Correctie/R_M_GR"
mask_path = "/Users/alexsamyn/Documents/BAP_(Mac)/Data Brussel/Raster_Buildings.tif"  # aangepaste masker: gebouwen = 1
output_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/Radio_Correctie/Output_Final_Try"
referentie_path = "/Users/alexsamyn/Documents/BAP_(Mac)/DV_Geo_Correctie/R_M_GR/R_20230405_GR.tif"

os.makedirs(output_folder, exist_ok=True)

def resample_mask_to_image(mask_path, referentie_path):
    with rasterio.open(referentie_path) as ref:
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

    return np.where(mask_resampled == 1, 0, 1).astype(np.uint8)

def process_image(path, mask, reference_img=None):
    with rasterio.open(path) as src:
        img = src.read()
        profile = src.profile

    img = np.transpose(img, (1, 2, 0))
    masked_img = img.copy()
    masked_img[mask == 0] = 0

    if reference_img is not None:
        valid_mask = (mask == 1) & (np.sum(img, axis=2) > 0)
        matched = masked_img.copy()
        for b in range(img.shape[2]):
            matched_band = match_histograms(
                masked_img[:, :, b][valid_mask],
                reference_img[:, :, b][valid_mask],
                channel_axis=None
            )
            temp = matched[:, :, b]
            temp[valid_mask] = matched_band
            matched[:, :, b] = temp
        masked_img = matched

    return np.transpose(masked_img, (2, 0, 1)), profile

def normalize(img):
    flat = img[~np.isnan(img)]
    p1, p99 = np.percentile(flat, [1, 99])
    return np.clip((img - p1) / (p99 - p1), 0, 1)

def main():
    mask = resample_mask_to_image(mask_path, referentie_path)

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

            if "20230301" in filename:
                original_img = rasterio.open(path).read()
                original_img = np.transpose(original_img, (1, 2, 0))
                original_img[mask == 0] = np.nan

                corrected_img = np.transpose(result_img, (1, 2, 0))
                corrected_img[mask == 0] = np.nan

                rgb_orig = normalize(np.stack((original_img[:, :, 5], original_img[:, :, 3], original_img[:, :, 1]), axis=2))
                rgb_corr = normalize(np.stack((corrected_img[:, :, 5], corrected_img[:, :, 3], corrected_img[:, :, 1]), axis=2))

                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
                axs[0].imshow(np.nan_to_num(rgb_orig))
                axs[0].set_title("Origineel RGB - R_20230301_GR.tif")
                axs[0].axis('off')
                axs[1].imshow(np.nan_to_num(rgb_corr))
                axs[1].set_title("Na Histogram Matching RGB - R_20230301_GR.tif")
                axs[1].axis('off')
                plt.tight_layout()
                plt.show()

                band_names = ["Coastal Blue", "Blue", "Green I", "Green", "Yellow", "Red", "Red Edge", "NIR"]
                band_colors = ["deepskyblue", "blue", "mediumseagreen", "green", "gold", "red", "darkred", "purple"]
                fig, axs = plt.subplots(4, 2, figsize=(24, 20))
                axs = axs.flatten()

                max_val = max(np.nanmax(original_img), np.nanmax(corrected_img))

                for i in range(8):
                    orig = original_img[:, :, i][mask == 1]
                    corr = corrected_img[:, :, i][mask == 1]
                    axs[i].hist(orig.ravel(), bins=256, color=band_colors[i], alpha=0.9, label="Origineel")
                    axs[i].hist(corr.ravel(), bins=256, color="orange", alpha=0.6, label="Gecorrigeerd")
                    
                    axs[i].set_title(f"{band_names[i]}")
                    axs[i].set_xlabel("Pixelwaarde")
                    axs[i].set_ylabel("Frequentie")
                    axs[i].set_xlim(-1000, max_val)
                    axs[i].legend()
                    axs[i].grid(True)
                    axs[i].margins(x=0.05)

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Fout bij verwerken {filename}: {e}")

        gc.collect()

if __name__ == "__main__":
    main()
