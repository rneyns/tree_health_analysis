import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

#Input & Output
input_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/GC"
mask_path = "/Users/alexsamyn/Documents/BAP_(Mac)/Data Brussel/Raster_Buildings_Null.tif"
output_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/RC"
referentie_path = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/GC/R_20230405_GC.tif"

os.makedirs(output_folder, exist_ok=True)

#Functies definiëren
def load_single_image(folder, target_filename):
    path = os.path.join(folder, target_filename)
    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
    return img, target_filename

def load_reference_image(path):
    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
        meta = src.meta
    print(f"Referentiebeeld geladen van: {path}")
    return img, meta

def resample_mask_to_image(mask_path, reference_image_path):
    with rasterio.open(reference_image_path) as ref:
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

    return (mask_resampled == 1)

def histogram_match_masked(image, reference, mask):
    matched = np.zeros_like(image)
    bands = image.shape[2]

    for b in range(bands):
        source_band = image[:, :, b]
        reference_band = reference[:, :, b]

        matched_band = match_histograms(
            source_band[mask], reference_band[mask], channel_axis=None
        )

        full_band = source_band.copy()
        full_band[mask] = matched_band
        matched[:, :, b] = full_band

    return matched.astype(image.dtype)

def save_multiband_image(array, path, reference_metadata_path):
    with rasterio.open(reference_metadata_path) as ref:
        profile = ref.profile

    profile.update({
        "count": array.shape[2],
        "dtype": array.dtype,
        "driver": "GTiff"
    })

    array = np.transpose(array, (2, 0, 1))

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)

def show_matching_visual(original, matched, reference, mask, filename=None, bands=(4, 3, 2)):
    def stretch(band):
        p2, p98 = np.percentile(band[mask], (2, 98))
        return np.clip((band - p2) / (p98 - p2), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [f'Origineel ({filename})', f'Gematched ({filename})', 'Referentie']
    imgs = [original, matched, reference]

    for ax, img, title in zip(axes, imgs, titles):
        rgb = np.stack([stretch(img[:, :, b - 1]) for b in bands], axis=-1)
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    bands_total = original.shape[2]
    for band in range(bands_total):
        plt.figure(figsize=(8, 4))
        plt.hist(original[:, :, band][mask].ravel(), bins=100, alpha=0.5, label='Origineel')
        plt.hist(matched[:, :, band][mask].ravel(), bins=100, alpha=0.5, label='Gematched')
        plt.hist(reference[:, :, band][mask].ravel(), bins=100, alpha=0.5, label='Referentie')
        plt.title(f"Histogram binnen masker (Band {band+1}) - {filename}")
        plt.xlabel("Pixelwaarde")
        plt.ylabel("Frequentie")
        plt.legend()
        plt.tight_layout()
        plt.show()

# === Hoofdproces ===

def process_all_images_with_mask(input_folder, mask_path, output_folder, reference_path):
    reference_image, _ = load_reference_image(reference_path)
    mask = resample_mask_to_image(mask_path, reference_path)

    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith(".tif"):
            continue

        print(f"Bezig met: {filename}")
        img, _ = load_single_image(input_folder, filename)
        corrected = histogram_match_masked(img, reference_image, mask)

        if filename == os.path.basename(referentie_path):
            show_matching_visual(img, corrected, reference_image, mask, filename=filename)

        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_RC.tif")
        save_multiband_image(corrected, output_path, reference_path)
        print(f"Gecorrigeerd beeld opgeslagen naar: {output_path}")

# === Uitvoeren ===

if __name__ == "__main__":
    process_all_images_with_mask(input_folder, mask_path, output_folder, referentie_path)

