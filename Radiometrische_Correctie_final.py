import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

#Input & Output
input_folder = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Alex PlanetScope'
mask_path = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/radiometric correction/buildings_mask.tif'
output_folder = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Alex PlanetScope corrected'
referentie_path = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Alex PlanetScope/R_20230405_GC.tif'

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

import numpy as np

def _cdf_mapping_from_masked_1d(source_band, reference_band, mask,
                                nbins=2048, clip_percent=(0.5, 99.5), sample_max=250_000):
    """
    Build a piecewise-linear mapping f such that CDF_source(x) ≈ CDF_ref(f(x)),
    but CDFs are computed *only* over masked pixels (buildings).
    Returns (src_grid, mapped_values) to be used with np.interp.
    """
    # Pull masked values
    svals = source_band[mask]
    rvals = reference_band[mask]

    # Optional down-sampling for speed (keeps CDF shape)
    if svals.size > sample_max:
        idx = np.random.choice(svals.size, sample_max, replace=False)
        svals = svals[idx]
    if rvals.size > sample_max:
        idx = np.random.choice(rvals.size, sample_max, replace=False)
        rvals = rvals[idx]

    # Robust clip ranges to avoid outliers skewing the mapping
    s_lo, s_hi = np.percentile(svals, clip_percent)
    r_lo, r_hi = np.percentile(rvals, clip_percent)

    # Build histograms on clipped ranges
    s_hist, s_edges = np.histogram(np.clip(svals, s_lo, s_hi), bins=nbins, range=(s_lo, s_hi), density=False)
    r_hist, r_edges = np.histogram(np.clip(rvals, r_lo, r_hi), bins=nbins, range=(r_lo, r_hi), density=False)

    # Convert to CDFs
    s_cdf = np.cumsum(s_hist).astype(float); s_cdf /= s_cdf[-1] if s_cdf[-1] > 0 else 1.0
    r_cdf = np.cumsum(r_hist).astype(float); r_cdf /= r_cdf[-1] if r_cdf[-1] > 0 else 1.0

    # Grid at bin centers
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # For each source CDF level, find reference intensity with same CDF
    # This yields a monotone piecewise-linear mapping s -> r
    mapped_vals = np.interp(s_cdf, r_cdf, r_centers)

    # Ensure strictly increasing grid (guard against flat tails)
    # Remove duplicate s_centers if present
    keep = np.concatenate(([True], np.diff(s_centers) > 0))
    return s_centers[keep], mapped_vals[keep]


def fit_luts_from_mask(source_img, reference_img, mask,
                       nbins=2048, clip_percent=(0.5, 99.5), sample_max=250_000):
    """
    Fit per-band LUTs using only masked (building) pixels.
    Returns list of (src_grid, mapped_values) per band.
    """
    if source_img.dtype.kind in ("u", "i"):
        src = source_img.astype(np.float32)
    else:
        src = source_img.copy()
    if reference_img.dtype.kind in ("u", "i"):
        ref = reference_img.astype(np.float32)
    else:
        ref = reference_img.copy()

    bands = src.shape[2]
    luts = []
    for b in range(bands):
        s_grid, r_map = _cdf_mapping_from_masked_1d(
            src[:, :, b], ref[:, :, b], mask,
            nbins=nbins, clip_percent=clip_percent, sample_max=sample_max
        )
        luts.append((s_grid, r_map))
    return luts


def apply_luts_to_image(image, luts, out_dtype=None):
    """
    Apply per-band LUTs globally (to all pixels in the image).
    Keeps dtype unless out_dtype provided.
    """
    if out_dtype is None:
        out_dtype = image.dtype
    imgf = image.astype(np.float32) if image.dtype.kind in ("u", "i") else image.copy()

    out = np.empty_like(imgf)
    for b, (s_grid, r_map) in enumerate(luts):
        band = imgf[:, :, b]
        # Piecewise-linear interpolation for every pixel
        out[:, :, b] = np.interp(band.ravel(), s_grid, r_map,
                                 left=r_map[0], right=r_map[-1]).reshape(band.shape)

    # If original was integer, round & clip to its valid range
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(out_dtype)
    else:
        out = out.astype(out_dtype)
    return out


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
        # 1) fit mapping only on buildings, 2) apply mapping to the whole image
        luts = fit_luts_from_mask(img, reference_image, mask, nbins=2048, clip_percent=(0.5, 99.5))
        corrected = apply_luts_to_image(img, luts)

        if filename == os.path.basename(referentie_path):
            show_matching_visual(img, corrected, reference_image, mask, filename=filename)

        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_RC.tif")
        save_multiband_image(corrected, output_path, reference_path)
        print(f"Gecorrigeerd beeld opgeslagen naar: {output_path}")

# === Uitvoeren ===

if __name__ == "__main__":
    process_all_images_with_mask(input_folder, mask_path, output_folder, referentie_path)

