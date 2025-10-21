"""
Image reprocessing script for dicom scans.
- Loads all .dcm files per patient
- Safely handles JPEG Lossless compression (using gdcm/pylibjpeg)
- Converts pixel data to Hounsfield Units (HU)
- Saves each patient’s CT volume as a .npy file
"""
# Import necessary libraries
from pathlib import Path
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

# Define base directories
BASE_DIR = Path(__file__).resolve().parent
SCANS_DIR = BASE_DIR / "data" / "train"
OUTPUT_DIR = BASE_DIR / "processed_scans_hu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Safe DICOM read function
def safe_dcm_read(dcm_path):
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        image = dcm.pixel_array  # may trigger decompression
    except Exception as e:
        print(f"Skipping {dcm_path.name}: {e}")
        return None

    # Apply VOI LUT (enhances contrast if present)
    try:
        image = apply_voi_lut(image, dcm)
    except Exception:
        pass

    image = image.astype(np.int16, copy=False)
    return image, dcm


# HU conversion function
def convert_to_hu(image, dcm):
    intercept = getattr(dcm, "RescaleIntercept", 0)
    slope = getattr(dcm, "RescaleSlope", 1)

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return image


# Main processing loop
def process_all_patients():
    for patient_dir in tqdm(sorted(SCANS_DIR.iterdir()), desc="Processing patients"):
        if not patient_dir.is_dir():
            continue

        slices = []
        for dcm_file in sorted(patient_dir.glob("*.dcm"), key=lambda x: int(x.stem)):
            result = safe_dcm_read(dcm_file)
            if result is None:
                continue

            image, dcm = result
            image_hu = convert_to_hu(image, dcm)
            slices.append(image_hu)

        if len(slices) == 0:
            print(f"⚠️ No valid slices for {patient_dir.name}. Skipping.")
            continue

        # Stack all slices into a single 3D volume
        volume_hu = np.stack(slices, axis=0)

        # Save as .npy file
        output_path = OUTPUT_DIR / f"{patient_dir.name}.npy"
        np.save(output_path, volume_hu)

        print(f"Saved {output_path.name} with shape {volume_hu.shape}")


if __name__ == "__main__":
    process_all_patients()
