import os
import argparse
import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology

def load_scan(patient_folder):
    """
    Loads DICOM slices from a folder and sorts them by InstanceNumber.
    """
    slices = [pydicom.dcmread(os.path.join(patient_folder, s)) for s in os.listdir(patient_folder)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices

def get_pixels_hu(slices):
    """
    Converts raw DICOM pixel values to Hounsfield Units (HU).
    Handles rescale slope and intercept.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    
    # Set outside-of-scan pixels to 0 (air is usually -1000)
    image[image == -2000] = 0
    
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def segment_lung_mask(image_hu):
    """
    Segments the lungs from the CT scan using HU thresholds and morphological operations.
    """
    segmented_mask = np.zeros_like(image_hu, dtype=np.uint8)
    
    for i, slice_img in enumerate(image_hu):
        # Step 1: Create a solid body mask (tissue > -500 HU)
        body_mask = slice_img > -500
        body_mask = binary_fill_holes(body_mask)
        
        # Step 2: Find air inside the body (approx -1000 to -400 HU)
        air_mask = (slice_img > -1000) & (slice_img < -400)
        internal_air_mask = air_mask & body_mask
        
        # Step 3: Keep the two largest air components (Lungs)
        labels = measure.label(internal_air_mask)
        props = measure.regionprops(labels)
        
        if len(props) < 1:
            continue

        # Sort regions by area
        sorted_labels_by_area = [p.label for p in sorted(props, key=lambda p: p.area, reverse=True)]
        
        lung_mask_slice = np.zeros_like(slice_img, dtype=np.uint8)
        
        # Add largest component
        lung_mask_slice[labels == sorted_labels_by_area[0]] = 1
        
        # Add second largest component if it exists (the other lung)
        if len(sorted_labels_by_area) > 1:
            lung_mask_slice[labels == sorted_labels_by_area[1]] = 1
            
        # Step 4: Cleanup
        lung_mask_slice = morphology.binary_closing(lung_mask_slice, footprint=morphology.disk(5))
        
        segmented_mask[i] = lung_mask_slice

    return segmented_mask

def select_lower_slices(image_stack, retention_ratio=0.55):
    """
    Selects the lower percentage of slices where fibrosis is often most prevalent.
    """
    total_slices = image_stack.shape[0]
    num_to_keep = int(np.round(total_slices * retention_ratio))
    
    # Slices are usually ordered top-to-bottom, so take from the end of the array
    start_index = total_slices - num_to_keep
    
    # Ensure we don't go out of bounds
    start_index = max(0, start_index)
    
    return image_stack[start_index:, :, :]

def prepare_for_cnn(image_stack, target_size=(256, 256)):
    """
    Resizes and normalizes the image stack for CNN input.
    Clip HU to [-1000, 400] and normalize to [0, 1].
    """
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    
    # Normalize
    image_stack = (image_stack - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image_stack[image_stack > 1] = 1.
    image_stack[image_stack < 0] = 0.
    
    resized_stack = []
    for i in range(image_stack.shape[0]):
        slice_img = image_stack[i, :, :]
        resized_img = cv2.resize(slice_img, target_size, interpolation=cv2.INTER_AREA)
        resized_stack.append(resized_img)
        
    # Add channel dimension (N, 256, 256, 1)
    return np.expand_dims(np.array(resized_stack), axis=-1)

def get_slope(patient_data):
    """
    Calculates the linear decline (slope) of FVC over weeks.
    """
    if len(patient_data) < 2:
        return np.nan
    try:
        fit = np.polyfit(patient_data['Weeks'], patient_data['FVC'], 1)
        return fit[0]
    except:
        return np.nan

def process_patient(patient_id, train_path, output_dir):
    """
    Pipeline to process a single patient and save result.
    """
    patient_folder = os.path.join(train_path, patient_id)
    save_path = os.path.join(output_dir, f"{patient_id}.npy")
    
    if os.path.exists(save_path):
        return True # Already processed
    
    try:
        # 1. Load
        slices = load_scan(patient_folder)
        
        # 2. To HU
        patient_hu = get_pixels_hu(slices)
        
        # 3. Segment Lungs
        lung_mask = segment_lung_mask(patient_hu)
        
        # 4. Apply Mask (Background becomes air: -2000)
        segmented_lungs = patient_hu.copy()
        segmented_lungs[lung_mask == 0] = -2000
        
        # 5. Select Slices
        selected_lungs = select_lower_slices(segmented_lungs)
        
        # 6. Resize & Normalize
        cnn_input = prepare_for_cnn(selected_lungs)
        
        # 7. Save
        if cnn_input.shape[0] > 0:
            np.save(save_path, cnn_input.astype(np.float32))
            return True
        
    except Exception as e:
        print(f"Failed to process {patient_id}: {e}")
        return False
    
    return False

def main():
    parser = argparse.ArgumentParser(description="OSIC Pulmonary Fibrosis Data Preprocessing")
    parser.add_argument("--data_dir", type=str, default="./", help="Root directory containing train.csv and train folder")
    parser.add_argument("--output_dir", type=str, default="./processed_train", help="Directory to save .npy files")
    parser.add_argument("--csv_name", type=str, default="train.csv", help="Name of the training CSV file")
    parser.add_argument("--meta_save_name", type=str, default="training_metadata.csv", help="Output name for processed metadata CSV")
    
    args = parser.parse_args()
    
    train_path = os.path.join(args.data_dir, "train")
    csv_path = os.path.join(args.data_dir, args.csv_name)
    
    if not os.path.exists(train_path) or not os.path.exists(csv_path):
        print(f"Error: Data not found at {args.data_dir}. Ensure 'train/' and '{args.csv_name}' exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Step 1: Metadata Processing ---
    print("Loading CSV and calculating target slopes...")
    train_df = pd.read_csv(csv_path)
    
    # Calculate slopes
    patient_slopes = train_df.groupby('Patient').apply(get_slope).reset_index()
    patient_slopes.columns = ['Patient', 'Slope']
    
    # Merge with baseline data
    baseline_df = train_df.drop_duplicates(subset=['Patient'], keep='first')
    meta_df = baseline_df.merge(patient_slopes, on='Patient').dropna()
    
    meta_save_path = os.path.join(args.data_dir, args.meta_save_name)
    meta_df.to_csv(meta_save_path, index=False)
    print(f"Metadata saved to {meta_save_path}. Total patients: {len(meta_df)}")
    
    # --- Step 2: Image Processing ---
    print(f"Starting image preprocessing. Saving to {args.output_dir}...")
    
    patients = meta_df['Patient'].unique()
    success_count = 0
    
    for patient_id in tqdm(patients):
        if process_patient(patient_id, train_path, args.output_dir):
            success_count += 1
            
    print(f"Done! Successfully processed {success_count}/{len(patients)} patients.")

if __name__ == "__main__":
    main()