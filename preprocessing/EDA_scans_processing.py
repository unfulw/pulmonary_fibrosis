import json
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
from skimage import exposure, filters
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA_DIR = REPO_ROOT / "data" / "train"
OUTPUT_DIR = REPO_ROOT / "data" / "processed_scans_hu"

# --------------------------------- #
# ------- dicom functions --------- #
# --------------------------------- # 

def read_patient_volume(patient_dir):
    # read all dicoms, collect with z-position for sorting
    slices = []
    for p in patient_dir.glob("*.dcm"):
        try:
            d = pydicom.dcmread(p, force=True)
            arr = d.pixel_array
        except Exception as e:
            print(f"skip {p.name}: {e}")
            continue

        # get z (fallback to InstanceNumber)
        ipp = getattr(d, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None else float(getattr(d, "SliceLocation", getattr(d, "InstanceNumber", 0)))
        slices.append((z, d, arr))

    if not slices:
        return None, None

    # sort by z ascending
    slices = sorted(slices, key=lambda x: x[0])
    ds_list = [t[1] for t in slices]
    images = np.stack([t[2] for t in slices], axis=0).astype(np.int16)

    # apply VOI LUT if present (apply to raw pixel data using associated dataset)
    try:
        images = np.stack([apply_voi_lut(d.pixel_array, d) for d in ds_list], axis=0).astype(np.int16)
    except Exception:
        pass

    # Convert to HU
    def to_hu(img, dcm):
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        img = img.astype(np.float64) * slope + intercept
        return img.astype(np.int16)

    hu_slices = np.stack([to_hu(img, d) for img, d in zip(images, ds_list)], axis=0)

    # spacing metadata
    px_spacing = getattr(ds_list[0], "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(ds_list[0], "SliceThickness", 1.0))
    spacing = (float(px_spacing[0]), float(px_spacing[1]), slice_thickness)

    meta = {
        "patient_id": patient_dir.name,
        "shape": hu_slices.shape,
        "spacing_mm": spacing
    }
    return hu_slices, meta


def load_all_patients(directory):
    data = {}
    for npy in directory.glob("*.npy"):
        patient_id = npy.stem
        vol = np.load(npy)
        
        meta_path = directory / f"{patient_id}.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        data[patient_id] = {
            "volume": vol,
            "meta": meta
        }
    return data


# --------------------------------- #
# ----- main script to run -------- #
# --------------------------------- # 

# Example save:
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    for patient_dir in tqdm(TRAIN_DATA_DIR.iterdir(), desc="Processing patients"):
        if not patient_dir.is_dir(): 
            continue
        vol, meta = read_patient_volume(patient_dir)
        if vol is None: 
            continue
        np.save(OUTPUT_DIR / f"{patient_dir.name}.npy", vol)
        with open(OUTPUT_DIR / f"{patient_dir.name}.json", "w") as f:
            json.dump(meta, f)
