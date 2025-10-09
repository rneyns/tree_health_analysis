#Script for radiometric correction of multispectral satellite images

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.exposure import match_histograms

#Input & Output
input_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/DV_Geo_Correctie/R_M_GR"
mask_path = "/Users/alexsamyn/Documents/BAP_(Mac)/Data Brussel/Raster_Buildings_Null.tif"
output_folder = "/Users/alexsamyn/Documents/BAP_(Mac)/Radio_Correctie/Output"
referentie_path = "/Users/alexsamyn/Documents/BAP_(Mac)/DV_Geo_Correctie/R_M_GR/R_20230405_GR.tif"

os.makedirs(output_folder, exist_ok=True)

#Functies definiëren

def create_histogram(raster, title):
    # Flatten and remove NaNs or nodata
    flat = raster.flatten()
    flat = flat[~np.isnan(flat)]

    # Create histogram
    plt.hist(flat, bins=100, density=True, alpha=0.6, color='gray')
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.show()

def load_images(folder):
    images = [] #Lege lijst met naam images waarin ik de latere numpy arrays opslaag
    filenames = [] #Lege lijst met naam filenames waarin de bestandnamen van tif worden bijgehouden
    
    #Loop over alle bestanden die gegeven worden en orden ze alfabetisch
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".tif"):
            path = os.path.join(folder, filename) #pathnaam = folder + filename --> rasterio heeft dit nodig
            with rasterio.open(path) as src: #openen van beelden en door with worden ze ook goed afgesloten
                img = src.read() #rasterdata lezen als numpy arrays (band, hoogte, breedte)
                img = np.transpose(img, (1, 2, 0)) #veranderen naar (hoogte, breedte, band) want handiger voor visualisatie waarbij de banden op het einde komen
            images.append(img) #getransformeerde beeld wordt toegevoegd aan de lijst images
            filenames.append(filename) #de bestandsnaam
    return images, filenames #lijst van ingelezen en getransformeerde beeldarrays en lijst met bestandnamen

#functie definieren naar maskerbestand gaan en referencie image (wat moet deze zijn?)
def resample_mask_to_image(mask_path, referentie_path):
    with rasterio.open(referentie_path) as ref:
        ref_profile = ref.profile #dictionary met metadata
        ref_shape = (ref.height, ref.width) #aantal rijen en kolommen 
        ref_transform = ref.transform #omzetten naar coordinaten
        ref_crs = ref.crs 

    with rasterio.open(mask_path) as src:
        mask_data = src.read(1) #2D-array met waarde 0 en 1 van de mask
        mask_resampled = np.empty(ref_shape, dtype=mask_data.dtype) #lege array met dezelfde afmetingen

        reproject(
            source=mask_data,
            destination=mask_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )

    return (mask_resampled == 0) #geeft aan waar de boolean True is = dus vegetatie is 

def load_reference_image(referentie_path):
    with rasterio.open(referentie_path) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
        meta = src.meta
    print(f"Referentiebeeld geladen van: {referentie_path}")
    return img, meta

# Laad alle beelden en referentie
images, filenames = load_images(input_folder)
reference_image, reference_meta = load_reference_image(referentie_path)
vegetation_mask = resample_mask_to_image(mask_path, referentie_path)

# Verwerk de eerste afbeelding als test
image = images[0]
matched = np.copy(image)

def histogram_match_masked(image, reference, mask):
    matched = np.copy(image)
    for b in range(image.shape[2]):
        create_histogram(reference_image[:, :, b][vegetation_mask], f"Reference histo band {b}")
        create_histogram(image[:, :, b][vegetation_mask], f"Original histo band {b}")
        matched_band = match_histograms(
            image[:, :, b][vegetation_mask],
            reference_image[:, :, b][vegetation_mask],
            channel_axis=None
            )
        create_histogram(matched[:, :, b][vegetation_mask], f"Matched histo band {b}")
        matched[:, :, b][vegetation_mask] = matched_band
    return matched

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

# Main proces
def process_images_with_mask(input_folder, mask_path, output_folder, referentie_path):
    images, filenames = load_images(input_folder)
    reference_image = load_reference_image(referentie_path)
    vegetation_mask = resample_mask_to_image(mask_path, referentie_path)

    # Zoek index van referentiebeeld zodat we die kunnen overslaan
    referentie_bestand = os.path.basename(referentie_path)
    try:
        ref_idx = filenames.index(referentie_bestand)
    except ValueError:
        ref_idx = -1  # Niet gevonden, dus niks wordt overgeslagen

    for i, (img, filename) in enumerate(zip(images, filenames)):
        if i == ref_idx:
            print(f"Referentiebeeld overgeslagen: {filename}")
            continue

        print(f"Verwerken: {filename}")
        corrected = histogram_match_masked(img, reference_image, vegetation_mask)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_HIST.tif")
        save_multiband_image(corrected, output_path, referentie_path)
        print(f"Opgeslagen: {output_path}")

# Uitvoeren van het script
if __name__ == "__main__":
    process_images_with_mask(input_folder, mask_path, output_folder, referentie_path)
