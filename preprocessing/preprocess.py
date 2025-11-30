import numpy as np
import pydicom
import scipy.ndimage as ndimage
import skimage.morphology as morphology
import cv2
import torch
import numpy
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from skimage import measure
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

def rescale_to_hu(slices) -> np.ndarray:
  """
  Convert DICOM pixel values to Hounsfield Units (HU).
  
  Args:
    dcm: DICOM file dataset
    
  Returns:
    Image array in Hounsfield Units
  """
  image = np.stack([s.pixel_array for s in slices])
  image = image.astype(np.int16)
  image[image == -2000] = 0

  for slice_number, slice in enumerate(slices):
    intercept = slice.RescaleIntercept
    slope = slice.RescaleSlope
    if slope != 1:
      image[slice_number] = (slope * image[slice_number]).astype(np.float64)
      image[slice_number] = image[slice_number].astype(np.int16)
    image[slice_number] += np.int16(intercept)
  
  return np.array(image, dtype=np.int16)

def segment_lung_mask_definitive(image_hu):
    """
    Definitive version:
    1. Create a solid mask of the patient's body.
    2. Find all air within that body mask.
    3. Keep the two largest air regions (the lungs).
    4. Clean up the final mask.
    """
    print("Lung")
    segmented_mask = np.zeros_like(image_hu, dtype=np.uint8)
    
    for i, slice_img in enumerate(image_hu):
        # Step 1: Create a solid body mask
        # A threshold of -500 HU is common for separating tissue from air
        body_mask = slice_img > -500
        # Fill holes in the body mask to make it solid
        body_mask = binary_fill_holes(body_mask)
        
        # Step 2: Find air inside the body
        # Threshold for air-like HU values
        air_mask = (slice_img > -1000) & (slice_img < -150)
        # Isolate the air that is *only* inside the body
        internal_air_mask = air_mask & body_mask
        
        # Step 3: Keep the two largest components
        labels = measure.label(internal_air_mask)
        props = measure.regionprops(labels)
        
        # for some top/bottom slices to not contain both lungs
        if len(props) < 1:
            continue

        areas = [prop.area for prop in props]
        sorted_labels_by_area = [p.label for p in sorted(props, key=lambda p: p.area, reverse=True)]
        
        lung_mask_slice = np.zeros_like(slice_img, dtype=np.uint8)
        # Add the largest component
        lung_mask_slice[labels == sorted_labels_by_area[0]] = 1
        # If there's a second large one, add it too
        if len(sorted_labels_by_area) > 1:
            lung_mask_slice[labels == sorted_labels_by_area[1]] = 1
            
        # Step 4: Final cleanup with morphological operations
        lung_mask_slice = morphology.binary_closing(lung_mask_slice, footprint=morphology.disk(7))
        
        segmented_mask[i] = lung_mask_slice

    return segmented_mask

def window_image_batch(images: torch.Tensor, window_center: int, window_width: int) -> torch.Tensor:
  """
  Apply CT windowing to batch of images.
  
  Args:
    images: Tensor of shape (N, 1, H, W) or (N, H, W)
    window_center: Window center value
    window_width: Window width value
    
  Returns:
      Binary mask: 1.0 if pixel in window range, 0.0 otherwise
  """
  img_min = window_center - window_width // 2
  img_max = window_center + window_width // 2
  
  # Create binary mask: True if in range, False otherwise
  in_window = (images >= img_min) & (images <= img_max)
  
  # Convert to float (1.0 or 0.0)
  windowed_images = in_window.float()
  
  return windowed_images

def gpu_dilation_batch(images: torch.Tensor, dilation_kernel_size: int) -> torch.Tensor:
  """
  Apply morphological dilation to batch of images.
  
  Args:
    images: Tensor of shape (N, 1, H, W)
    dilation_kernel_size: Size of dilation kernel
    
  Returns:
    Dilated images with same shape as input
  """
  kernel = torch.ones(1, 1, dilation_kernel_size, dilation_kernel_size, dtype=torch.float32, device=images.device)
  padding = (dilation_kernel_size - 1) // 2
  
  dilated = F.conv2d(images, kernel, padding=padding)
  
  # Restore dimensions if kernel size is even
  if dilation_kernel_size % 2 == 0:
    dilated = F.pad(dilated, (0, 1, 0, 1))
  
  return dilated

def mask_scans(images: list[np.ndarray], window_level: int = 40, window_width: int = 80) -> list[np.ndarray]:
  """
  Generate masks for batch of CT scans using GPU acceleration.
  
  Args:
    images: List of numpy arrays, each of shape (H, W)
    window_level: CT window center (default -600 for lung)
    window_width: CT window width (default 1500 for capture full lung range)
    
  Returns:
    List of binary masks, same length as input
  """
  if len(images) == 0:
    return []
  
  # Stack images into batch tensor (N, 1, H, W)
  images_batch = torch.stack([
    torch.tensor(img, dtype=torch.float32, device=device) 
    for img in images
  ]).unsqueeze(1)
  
  # Apply windowing on batch
  thresholded_batch = window_image_batch(images_batch, window_level, window_width)
  
  # Morphological dilation with 4x4 kernel on batch
  segmentation_batch = gpu_dilation_batch(thresholded_batch, 4)
  
  # Connected components must be done per-image (no batch operation available)
  masks = []
  for i in range(len(images)):
    segmentation = segmentation_batch[i, 0].cpu().numpy()
    
    # Find largest connected component
    labels, label_nb = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0
    mask = labels == label_count.argmax()
    
    masks.append(mask)
  
  # Stack masks back to batch tensor for GPU operations
  masks_batch = torch.stack([
    torch.tensor(mask, dtype=torch.float32, device=device) 
    for mask in masks
  ]).unsqueeze(1)
  
  # Apply 1x1 dilation (no-op for odd kernel, but kept for compatibility)
  masks_batch = gpu_dilation_batch(masks_batch, 1)
  
  # Binary fill holes - must be done per-image
  masks_filled_list = []
  for i in range(len(masks)):
    mask_cpu = masks_batch[i, 0].cpu().numpy()
    mask_filled = ndimage.binary_fill_holes(mask_cpu)
    masks_filled_list.append(mask_filled)
  
  # Stack back to batch for final dilation
  masks_batch = torch.stack([
    torch.tensor(mask, dtype=torch.float32, device=device) 
    for mask in masks_filled_list
  ]).unsqueeze(1)
  
  # Apply 3x3 dilation on batch
  masks_batch = gpu_dilation_batch(masks_batch, 3)
  
  # Move masks to CPU
  masks_batch = masks_batch.cpu()

  # Convert back to numpy list
  final_masks = [masks_batch[i, 0].numpy() for i in range(len(images))]
  
  return final_masks

def preprocess_dicom(patient_id: str, dcms: list[pydicom.dataset.FileDataset], size: int = 256) -> list[np.ndarray]:
  """
  Preprocess batch of DICOM files: HU conversion, masking, and resizing.
  
  Args:
    dcms: List of DICOM file datasets
    
  Returns:
    List of preprocessed 256x256 scans
  """
  # Convert to Hounsfield units
  hu_scans = rescale_to_hu(dcms)
  
  # Generate masks using GPU batch processing
  masks = mask_scans(hu_scans)
  
  # Apply masks and resize
  preprocessed_scans = []
  for hu_scan, mask in zip(hu_scans, masks):
    # Apply lung windowing: [-1000, 400] HU range for lung tissue
    windowed_scan = np.clip(hu_scan, -1000, 400)
    
    # Normalize to [0, 1] range
    normalized_scan = (windowed_scan - (-1000)) / (400 - (-1000))
    
    # Apply mask (background becomes 0, lung tissue is in [0, 1])
    masked_scan = np.float32(normalized_scan * mask)
    resized_scan = cv2.resize(masked_scan, (size, size), interpolation=cv2.INTER_AREA)
    preprocessed_scans.append(resized_scan)
  
  return np.array(preprocessed_scans, dtype=np.float32)

def preprocess_lung_segmentation(patient_id: str, dcms: list[pydicom.dataset.FileDataset], size: int = 256) -> list[np.ndarray]:
  """
  Preprocess batch of DICOM files: HU conversion, masking, and resizing.
  
  Args:
    dcms: List of DICOM file datasets
    
  Returns:
    List of preprocessed scans
  """
  # Convert to Hounsfield units
  hu_scans = rescale_to_hu(dcms)

  body_mask = mask_scans(hu_scans, window_level=0, window_width=300)
  body_mask = [np.array(mask, dtype=np.int16) for mask in body_mask]

  hu_scans_masked = hu_scans.copy()
  hu_scans_masked[np.array(body_mask) == 0] = -2000

  lung_mask = segment_lung_mask_definitive(hu_scans_masked)
  
  segmented_lungs_hu = hu_scans_masked.copy()
  segmented_lungs_hu[lung_mask == 0] = -2000
  
  # Apply lung windowing: [-1000, 400] HU range for lung tissue
  windowed_scan = np.clip(segmented_lungs_hu, -1000, 400)
  
  # Normalize to [0, 1] range
  normalized_scan = (windowed_scan - (-1000)) / (400 - (-1000))
  
  # Apply mask (background becomes 0)
  normalized_scan[lung_mask == 0] = 0
  
  new_size = (size, size)
  normalized_scan = normalized_scan.squeeze()
  resized_scan = cv2.resize(normalized_scan, new_size, interpolation=cv2.INTER_LINEAR)

  return np.array(resized_scan, dtype=np.float32)

def get_test_preprocessed_scan(data_path: str, patient_id: str, scan_idx: int) -> np.ndarray:
  if os.path.exists(os.path.join(data_path, 'test_preprocessed_scans', patient_id, f'{scan_idx}.npy')):
    return np.load(os.path.join(data_path, 'test_preprocessed_scans', patient_id, f'{scan_idx}.npy'))

  try:
    dcm = pydicom.dcmread(os.path.join(data_path, 'test', patient_id, f'{scan_idx}.dcm'))
    # Check if pixel_array is readable
    pixel_array = dcm.pixel_array
  except Exception as e:
    # print(f'Error reading {os.path.join(data_path, 'test', patient_id, f'{scan_idx}.dcm')}: {e}')
    return None
  preprocessed_scan = preprocess_dicom(patient_id, [dcm])

  if not os.path.exists(os.path.join(data_path, 'test_preprocessed_scans', patient_id)):
    os.makedirs(os.path.join(data_path, 'test_preprocessed_scans', patient_id))
  np.save(os.path.join(data_path, 'test_preprocessed_scans', patient_id, f'{scan_idx}.npy'), preprocessed_scan)
  return preprocessed_scan

def get_preprocessed_scan(data_path: str, patient_id: str, scan_idx: int, lung_segmentation: bool = False, size: int = 256) -> np.ndarray:
  """
  Preprocess a single scan and save it to the preprocessed_scans folder
  If the scan is already preprocessed, load it from the preprocessed_scans folder
  If the scan is not preprocessed, preprocess it and save it to the preprocessed_scans folder
  Input: data_path: str, patient_id: str, scan_idx: int
  Returns: preprocessed_scan: np.ndarray
  """

  if not lung_segmentation and os.path.exists(os.path.join(data_path, f'preprocessed_scans_{size}', patient_id, f'{scan_idx}.npy')):
    return np.load(os.path.join(data_path, f'preprocessed_scans_{size}', patient_id, f'{scan_idx}.npy'))
  elif lung_segmentation:
    return np.load(os.path.join(data_path, f'preprocessed_lung_segmentation_{size}', patient_id, f'{scan_idx}.npy'))
  
  if not os.path.exists(os.path.join(data_path, 'train', patient_id, f'{scan_idx}.dcm')):
    return None
  
  try:
    # print("Redoing work")
    dcm = pydicom.dcmread(os.path.join(data_path, 'train', patient_id, f'{scan_idx}.dcm'))
    # Check if pixel_array is readable
    pixel_array = dcm.pixel_array
  except Exception as e:
    # print(f'Error reading {os.path.join(data_path, 'train', patient_id, f'{scan_idx}.dcm')}: {e}')
    return None
    
  if lung_segmentation:
    preprocessed_scan = preprocess_lung_segmentation(patient_id, [dcm], size)
    if not os.path.exists(os.path.join(data_path, f'preprocessed_lung_segmentation_{size}', patient_id)):
      os.makedirs(os.path.join(data_path, f'preprocessed_lung_segmentation_{size}', patient_id))
    np.save(os.path.join(data_path, f'preprocessed_lung_segmentation_{size}', patient_id, f'{scan_idx}.npy'), preprocessed_scan)
    return preprocessed_scan
  else:
    preprocessed_scan = preprocess_dicom(patient_id, [dcm], size)
    if not os.path.exists(os.path.join(data_path, f'preprocessed_scans_{size}', patient_id)):
      os.makedirs(os.path.join(data_path, f'preprocessed_scans_{size}', patient_id))
    np.save(os.path.join(data_path, f'preprocessed_scans_{size}', patient_id, f'{scan_idx}.npy'), preprocessed_scan)
    return preprocessed_scan


def preprocess_scans(data_path: str, lung_segmentation: bool = False, size: int = 256) -> dict[str, np.ndarray]:
  preprocessed_scans = dict()
  for patient_id in tqdm(os.listdir(os.path.join(data_path, 'train'))):
    patient_scans = []
    for scan_idx in range(1, len(os.listdir(os.path.join(data_path, 'train', patient_id))) + 1):
      scan = get_preprocessed_scan(data_path, patient_id, scan_idx, lung_segmentation, size)
      if scan is not None:
        patient_scans.append(scan)
    patient_scans = np.array(patient_scans, dtype=np.float32)
    preprocessed_scans[patient_id] = patient_scans
  return preprocessed_scans



if __name__ == "__main__":
  import os
  preprocess_scans('C:/Coding/pulmonary_fibrosis/osic-pulmonary-fibrosis-progression')
  # # Load DICOM files for testing
  # patient_dir = 'C:/Coding/pulmonary_fibrosis/osic-pulmonary-fibrosis-progression/'
  # scan = get_preprocessed_scan(patient_dir, 'ID00007637202177411956430', 15)
  # plt.imshow(scan.squeeze(), cmap='gray')
  # plt.show()