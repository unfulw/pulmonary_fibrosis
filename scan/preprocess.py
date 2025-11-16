import numpy as np
import pydicom
import scipy.ndimage as ndimage
import skimage.morphology as morphology
import cv2
import torch
import numpy
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

def rescale_to_hu(patient_id: str, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
  """
  Convert DICOM pixel values to Hounsfield Units (HU).
  
  Args:
    dcm: DICOM file dataset
    
  Returns:
    Image array in Hounsfield Units
  """
  # Get rescale parameters (default to identity transform if missing)
  # Print log if intercept or slope is not found, default to 0 and 1 if not found
  if not hasattr(dcm, 'RescaleIntercept'):
    print(f'RescaleIntercept not found for {patient_id}')
  if not hasattr(dcm, 'RescaleSlope'):
    print(f'RescaleSlope not found for {patient_id}')
  intercept = getattr(dcm, 'RescaleIntercept', 0)
  slope = getattr(dcm, 'RescaleSlope', 1)

  # Convert to HU
  hu_image = dcm.pixel_array * slope + intercept
  
  return hu_image

def window_image_batch(images: torch.Tensor, window_center: int, window_width: int) -> torch.Tensor:
  """
  Apply CT windowing to batch of images.
  
  Args:
    images: Tensor of shape (N, 1, H, W) or (N, H, W)
    window_center: Window center value
    window_width: Window width value
    
  Returns:
    Windowed images with same shape as input
  """
  img_min = window_center - window_width // 2
  img_max = window_center + window_width // 2
  windowed_images = torch.clamp(images, img_min, img_max)
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
    window_level: CT window center (default 40 for bone)
    window_width: CT window width (default 80)
    
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

def preprocess_dicom(patient_id: str, dcms: list[pydicom.dataset.FileDataset]) -> list[np.ndarray]:
  """
  Preprocess batch of DICOM files: HU conversion, masking, and resizing.
  
  Args:
    dcms: List of DICOM file datasets
    
  Returns:
    List of preprocessed 256x256 scans
  """
  # Convert to Hounsfield units
  hu_scans = [rescale_to_hu(patient_id, dcm) for patient_id, dcm in zip([patient_id] * len(dcms), dcms)]
  
  # Generate masks using GPU batch processing
  masks = mask_scans(hu_scans)
  
  # Apply masks and resize
  preprocessed_scans = []
  for hu_scan, mask in zip(hu_scans, masks):
    masked_scan = np.float32(hu_scan * mask)
    resized_scan = cv2.resize(masked_scan, (256, 256), interpolation=cv2.INTER_AREA)
    preprocessed_scans.append(resized_scan)
  
  return np.array(preprocessed_scans, dtype=np.float32)

def get_test_preprocessed_scan(data_path: str, patient_id: str, scan_idx: int) -> np.ndarray:
  if os.path.exists(os.path.join(data_path, 'test_preprocessed_scans', patient_id, f'{scan_idx}.npy')):
    return np.load(os.path.join(data_path, 'test_preprocessed_scans', patient_id, f'{scan_idx}.npy'))

  try:
    print("Redoing work")
    dcm = pydicom.dcmread(os.path.join(data_path, 'test', patient_id, f'{scan_idx}.dcm'))
    # Check if pixel_array is readable
    pixel_array = dcm.pixel_array
  except Exception as e:
    print(f'Error reading {os.path.join(data_path, 'test', patient_id, f'{scan_idx}.dcm')}: {e}')
    return None
  preprocessed_scan = preprocess_dicom(patient_id, [dcm])

  if not os.path.exists(os.path.join(data_path, 'test_preprocessed_scans', patient_id)):
    os.makedirs(os.path.join(data_path, 'test_preprocessed_scans', patient_id))
  np.save(os.path.join(data_path, 'test_preprocessed_scans', patient_id, f'{scan_idx}.npy'), preprocessed_scan)
  return preprocessed_scan

def get_preprocessed_scan(data_path: str, patient_id: str, scan_idx: int) -> np.ndarray:
  """
  Preprocess a single scan and save it to the preprocessed_scans folder
  If the scan is already preprocessed, load it from the preprocessed_scans folder
  If the scan is not preprocessed, preprocess it and save it to the preprocessed_scans folder
  Input: data_path: str, patient_id: str, scan_idx: int
  Returns: preprocessed_scan: np.ndarray
  """
  if os.path.exists(os.path.join(data_path, 'preprocessed_scans', patient_id, f'{scan_idx}.npy')):
    return np.load(os.path.join(data_path, 'preprocessed_scans', patient_id, f'{scan_idx}.npy'))
  
  try:
    print("Redoing work")
    dcm = pydicom.dcmread(os.path.join(data_path, 'train', patient_id, f'{scan_idx}.dcm'))
    # Check if pixel_array is readable
    pixel_array = dcm.pixel_array
  except Exception as e:
    print(f'Error reading {os.path.join(data_path, 'train', patient_id, f'{scan_idx}.dcm')}: {e}')
    return None
  preprocessed_scan = preprocess_dicom(patient_id, [dcm])

  if not os.path.exists(os.path.join(data_path, 'preprocessed_scans', patient_id)):
    os.makedirs(os.path.join(data_path, 'preprocessed_scans', patient_id))
  np.save(os.path.join(data_path, 'preprocessed_scans', patient_id, f'{scan_idx}.npy'), preprocessed_scan)
  return preprocessed_scan

def preprocess_scans(data_path: str) -> dict[str, np.ndarray]:
  preprocessed_scans = dict()
  for patient_id in os.listdir(os.path.join(data_path, 'train')):
    patient_scans = []
    for scan_idx in range(1, len(os.listdir(os.path.join(data_path, 'train', patient_id))) + 1):
      scan = get_preprocessed_scan(data_path, patient_id, scan_idx)
      if scan is not None:
        patient_scans.append(scan)
    patient_scans = np.array(patient_scans, dtype=np.float32)
    preprocessed_scans[patient_id] = patient_scans
  return preprocessed_scans

if __name__ == "__main__":
  import time
  import matplotlib.pyplot as plt
  import os
  
  # Load DICOM files for testing
  patient_dir = '../osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/'
  num_files = len(os.listdir(patient_dir))
  dcms = [pydicom.dcmread(f'{patient_dir}{i}.dcm') for i in range(1, num_files + 1)]

  # Process as batch
  start_time = time.time()
  results = preprocess_dicom(dcms)
  end_time = time.time()
  
  # Display result
  plt.imshow(results[15], cmap='gray')
  plt.title('Preprocessed CT Scan')
  plt.show()
  
  print(f'Preprocessing time: {end_time - start_time:.4f} seconds')
  print(f'Output shape: {results[0].shape}')
  print(f'Processed {len(results)} scans')