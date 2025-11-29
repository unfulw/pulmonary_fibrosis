import cv2
import numpy as np
import pandas as pd
import os 
import torch
import torch.nn.functional as F
import torchvision
import random
import pydicom
import os
import pandas as pd
import sys
import pickle

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from preprocessing.scan.preprocess import preprocess_dicom

data_dir = 'C:/Coding/pulmonary_fibrosis/osic-pulmonary-fibrosis-progression'
checkpoints_dir = 'C:/Coding/pulmonary_fibrosis/models/checkpoints'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

idx_to_feature_name = ['edge', 'corner', 'blob', 'lbp', 'gabor']

def get_idx_to_feature_name(idx):
  return idx_to_feature_name[idx]

# Input: List of patient_id
# Output: Tuple of (train_patient_ids, val_patient_ids)
# train_ratio: Ratio of training set
def train_val_split(patients: pd.DataFrame, train_ratio: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
  patient_ids = patients['Patient'].unique()
  random.shuffle(patient_ids)

  train_patients = patients[patients['Patient'].isin(patient_ids[:int(len(patient_ids) * train_ratio)])]
  val_patients = patients[patients['Patient'].isin(patient_ids[int(len(patient_ids) * train_ratio):])]

  return train_patients, val_patients

# Edge detection using Sobel operator
def extract_edge(image_array: np.ndarray, batching: bool = False) -> list[np.ndarray]:
  if device == torch.device('cuda'):
    batch = torch.tensor(image_array, dtype=torch.float32, device=device)
    # Convert to batch tensor (N, 1, H, W)
    if not batching:
      batch = batch.unsqueeze(0).unsqueeze(0)

    # Define Sobel kernels
    sobel_x = torch.tensor([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([
      [-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Apply Sobel filters to entire batch
    edges_x = F.conv2d(batch, sobel_x, padding=1)
    edges_y = F.conv2d(batch, sobel_y, padding=1)
    
    # Compute magnitude
    edges = 0.5 * torch.abs(edges_x) + 0.5 * torch.abs(edges_y)
    
    # Convert back to list of numpy arrays
    edges = edges.cpu().numpy()
    if not batching:
      edges = edges[0, 0]
    else:
      edges = edges[:, 0]

    del batch, edges_x, edges_y
    
    return edges
    
  edgesX = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
  edgesY = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
  absEdgesX = cv2.convertScaleAbs(edgesX)
  absEdgesY = cv2.convertScaleAbs(edgesY)
  edges = cv2.addWeighted(absEdgesX, 0.5, absEdgesY, 0.5, 0)
  return edges

def extract_corner(image_array: np.ndarray, block_size=2, ksize=3, k=0.04, batching: bool = False) -> list[np.ndarray]:
  if device == torch.device('cuda'):
    batch = torch.tensor(image_array, dtype=torch.float32, device=device)
    
    # Convert to batch tensor (N, 1, H, W)
    if not batching:
      batch = batch.unsqueeze(0).unsqueeze(0)
    
    # Sobel kernels
    sobel_x = torch.tensor([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, ksize, ksize)
    
    sobel_y = torch.tensor([
      [-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]
    ], dtype=torch.float32, device=device).view(1, 1, ksize, ksize)
    
    # Compute gradients for entire batch
    Ix = F.conv2d(batch, sobel_x, padding=1)
    Iy = F.conv2d(batch, sobel_y, padding=1)
    
    # Derivative products
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Gaussian kernel
    sigma = block_size / 2.0
    kernel_size = block_size * 2 + 1
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gaussian_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d.view(-1, 1) * gaussian_1d.view(1, -1)
    gaussian_2d = gaussian_2d.view(1, 1, kernel_size, kernel_size)
    
    # Apply Gaussian weighting
    padding = kernel_size // 2
    Sxx = F.conv2d(Ixx, gaussian_2d, padding=padding)
    Syy = F.conv2d(Iyy, gaussian_2d, padding=padding)
    Sxy = F.conv2d(Ixy, gaussian_2d, padding=padding)
    
    # Harris response for entire batch
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    harris_response = det - k * (trace * trace)
    
    # Apply threshold per image
    corners_list = []
    for i in range(harris_response.shape[0]):
      response = harris_response[i, 0]
      threshold = 0.05 * response.max()
      corners = response > threshold
      corners = corners.cpu().numpy().astype(np.float32)
      corners_list.append(corners)
    corners_list = np.array(corners_list)

    del batch, harris_response, Sxx, Syy, Sxy, Ixx, Iyy, Ixy, Ix, Iy
    if not batching:
      return corners_list[0]
    else:
      return corners_list
  
  # Apply cornerHarris
  score = cv2.cornerHarris(image_array, block_size, ksize, k)
  # 0.05 is a threshold to determine if a corner is valid
  corners = score > 0.05 * score.max()

  # Could consider returning original score without thresholding
  return corners

def extract_blob(image_array, batching: bool = False):
  if device == torch.device('cuda'):
    batch = torch.tensor(image_array, dtype=torch.float32, device=device)
    
    # Convert to batch tensor (N, 1, H, W)
    if not batching:
      batch = batch.unsqueeze(0).unsqueeze(0)

    # Apply blurs to entire batch
    blurred1 = torchvision.transforms.functional.gaussian_blur(batch, kernel_size=(5, 5), sigma=(0.8, 0.8))
    blurred2 = torchvision.transforms.functional.gaussian_blur(batch, kernel_size=(11, 11), sigma=(1.8, 1.8))
    
    # Compute DoG for entire batch
    dog = blurred1 - blurred2
    
    # Normalize each image separately
    results = []
    for i in range(dog.shape[0]):
      dog_single = dog[i, 0]
      dog_min = dog_single.min()
      dog_max = dog_single.max()
      dog_normalized = (dog_single - dog_min) / (dog_max - dog_min + 1e-8) * 255
      results.append(dog_normalized.cpu().numpy())
    
    del batch, blurred1, blurred2, dog, dog_single, dog_min, dog_max, dog_normalized
    results = np.array(results)
    if not batching:
      return results[0]
    else:
      return results

  # Use DoG (Difference of Gaussian) to detect blobs
  # First, apply GaussianBlur to the image
  blurred1 = cv2.GaussianBlur(image_array, (5, 5), 0.8)
  blurred2 = cv2.GaussianBlur(image_array, (11, 11), 1.8)

  # Then, compute the difference between the blurred image and the original image
  dog = cv2.subtract(blurred1, blurred2)

  # minmax normalize the result to enhance contrast
  dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

  return dog

def extract_local_binary_pattern(image_array, batching: bool = False):
  if device == torch.device('cuda'):
    image = torch.tensor(image_array, dtype=torch.float32, device=device)
    
    # Convert to batch tensor (N, 1, H, W)
    if not batching:
      image = image.unsqueeze(0).unsqueeze(0)

    # Create kernels for extracting each neighbor
    neighbor_kernels = []
    positions = [
      (0, 0), (0, 1), (0, 2),  # Top row
      (1, 2),                   # Right
      (2, 2), (2, 1), (2, 0),  # Bottom row
      (1, 0)                    # Left
    ]
    
    for i, (row, col) in enumerate(positions):
      kernel = torch.zeros(1, 1, 3, 3, device=device)
      kernel[0, 0, row, col] = 1.0
      neighbor_kernels.append(kernel)
    
    # Extract center pixel value
    center_kernel = torch.zeros(1, 1, 3, 3, device=device)
    center_kernel[0, 0, 1, 1] = 1.0
    center = F.conv2d(image, center_kernel, padding=1)
  
    # Initialize LBP
    lbp_image = torch.zeros_like(center)
    
    # Compare each neighbor
    for k, kernel in enumerate(neighbor_kernels):
      neighbor = F.conv2d(image, kernel, padding=1)
      comparison = (neighbor >= center).float()
      lbp_image += comparison * (2 ** k)
    
    result = lbp_image.cpu().numpy()

    del image, center, center_kernel, neighbor_kernels, neighbor, comparison, lbp_image

    if not batching:
      return result[0][0]
    else:
      return result.squeeze(1)

  # Use CPU otherwise
  # Get image dimensions
  height, width = image_array.shape

  # Initialize LBP image
  lbp_image = np.zeros_like(image_array)

  # Define 8-neighborhood offsets (8-connected)
  offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

  # Apply LBP to each pixel (excluding border pixels)
  for i in range(1, height - 1):
    for j in range(1, width - 1):
      center_pixel = image_array[i, j]
      binary_pattern = 0

      # Compare center pixel with 8 neighbors
      for k, (di, dj) in enumerate(offsets):
        neighbor_pixel = image_array[i + di, j + dj]
        if neighbor_pixel >= center_pixel:
          binary_pattern |= (1 << k)

      lbp_image[i, j] = binary_pattern

  return lbp_image

def extract_gabor_feature(image_array, batching: bool = False):
  # Define Gabor filter parameters
  orientations = [0, 45, 90, 135]  # 4 orientations in degrees
  frequencies = [0.1, 0.2, 0.3]   # 3 different frequencies
  sigma = 2.0  # Standard deviation of Gaussian envelope
  gamma = 0.5  # Spatial aspect ratio

  # Initialize feature maps
  gabor_features = []

  if device == torch.device('cuda') and not batching:
    image_array = torch.tensor(image_array, dtype=torch.float32, device=device)
    image_array = image_array.unsqueeze(0).unsqueeze(0)
  elif device == torch.device('cuda'):
    image_array = torch.tensor(image_array, dtype=torch.float32, device=device)

  # Apply Gabor filters with different orientations and frequencies
  for orientation in orientations:
    for frequency in frequencies:
      # Convert orientation to radians
      theta = np.radians(orientation)

      # Create Gabor kernel
      kernel = cv2.getGaborKernel(
        ksize=(15, 15),  # Kernel size
        sigma=sigma,
        theta=theta,
        lambd=1.0/frequency,  # Wavelength
        gamma=gamma,
        psi=0,  # Phase offset
        ktype=cv2.CV_32F
      )

      if device == torch.device('cuda'):
        kernel = torch.tensor(kernel, dtype=torch.float32, device=device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        filtered = F.conv2d(image_array, kernel, padding=7)
        magnitude = torch.abs(filtered)
        
        min_val = magnitude.min()
        max_val = magnitude.max()

        # Apply the min-max normalization formula
        normalized_data = (magnitude - min_val) / (max_val - min_val)

        gabor_feature = normalized_data.cpu().numpy()
        del magnitude, min_val, max_val, normalized_data, filtered
        gabor_features.append(gabor_feature)

      else:
        # Apply filter
        filtered = cv2.filter2D(image_array, cv2.CV_32F, kernel)

        # Take magnitude of response
        magnitude = np.abs(filtered)

        # Normalize to 0-255 range
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        gabor_features.append(magnitude)

  # Combine all Gabor responses (mean of all filter responses)
  combined_gabor = np.mean(gabor_features, axis=0)

  if not batching:
    return combined_gabor[0][0]
  return combined_gabor

idx_to_feature_extractor = [extract_edge, extract_corner, extract_blob, extract_local_binary_pattern, extract_gabor_feature]

def read_dicom(path):
  dicom = pydicom.dcmread(path)
  # Convert image to float32
  image_array = np.float32(dicom.pixel_array)
  return image_array

def organize_dicom():
  # Reorganize numbering for dicom files in each patient folder
  for patient_id in os.listdir(f'{data_dir}/train'):
    # Rename each file into according order
    existing_files = os.listdir(f'{data_dir}/train/{patient_id}/')
    existing_files.sort(key=lambda x: int(x.split('.')[0]))
    for i, file_name in enumerate(existing_files):
      os.rename(f'{data_dir}/train/{patient_id}/{file_name}', f'{data_dir}/train/{patient_id}/{i+1}.dcm')

def load_patient_scans(patient_id, scan_batch_size: int = 12):
  if not os.path.exists(f'{data_dir}/preprocessed_scans/{patient_id}'):
    return None
  patient_scan_count = len(os.listdir(f'{data_dir}/preprocessed_scans/{patient_id}'))
  skip_size = patient_scan_count // scan_batch_size
  remainder = patient_scan_count % scan_batch_size
  scans = []
  curr = 0
  while curr < patient_scan_count:
    scan = np.load(f'{data_dir}/preprocessed_scans/{patient_id}/{curr+1}.npy')
    # If scan dim is h*w, add channel dimension
    if len(scan.shape) == 2:
      scan = scan.reshape(1, scan.shape[0], scan.shape[1])
    scans.append(scan)
    curr += skip_size

    # Ensure exactly #scan_batch_size number of scans
    if remainder > 0:
      remainder -= 1
      curr += 1

    # if scans[-1].shape[1] != 512:
    #   scan = scans[-1]
    #   scan = scan.squeeze()
    #   resized_scan = cv2.resize(scan, (512, 512), interpolation=cv2.INTER_LINEAR)
    #   resized_scan = resized_scan.reshape(1, 512, 512)
    #   scans[-1] = resized_scan
    #   # Overwrite the npy file
    #   np.save(f'{data_dir}/preprocessed_scans/{patient_id}/{curr+1}.npy', scans[-1])
    
  return np.array(scans)

def extract_features(scan, batching: bool = False):
  features = []
  with torch.no_grad():
    for i in range(len(idx_to_feature_extractor)):
      features.append(idx_to_feature_extractor[i](scan, batching=batching))
  return np.array(features)

def get_pcas(x_train):
  pcas = []
  pca_component_count = [3500, 4000, 3000, 3000, 1000]

  for i in range(5):
    if os.path.exists(f'{data_dir}/trained_model/pca_feat{i}.pkl'):
      with open(f'{data_dir}/trained_model/pca_feat{i}.pkl', 'rb') as file:
        pcas.append(pickle.load(file))
      continue

    features = []
    for patient_id in tqdm(x_train.keys()):
      patient_scans = load_patient_scans(patient_id, scan_batch_size=64)
      if patient_scans is None:
        continue
      feature = idx_to_feature_extractor[i](patient_scans, batching=True)
      feature = [f.flatten() for f in feature]
      features += feature

    features = np.array(features)
    component_cnt = pca_component_count[i]
    pca = PCA(n_components=component_cnt)
    pca.fit(features)
    print(f"PCA explained variance, component count {pca.n_components_} : {pca.explained_variance_ratio_.sum()}")
    with open(f'{data_dir}/trained_model/pca_feat{i}.pkl', 'wb') as file:
      pickle.dump(pca, file)
    pcas.append(pca)
  
  return pcas

def get_rf_regressors(pcas, x_train, y_train):
  # Initialize random forest regressor for each feature type
  rf_regressors_n_estimators = [100, 40, 600, 500, 400]
  rf_regressors = [
      RandomForestRegressor(n_estimators=100, random_state=42)
      for _ in range(len(idx_to_feature_name))
  ]

  for i in range(5):
    features = []
    targets = []
    for patient_id in tqdm(x_train.keys()):
      patient_scans = load_patient_scans(patient_id, scan_batch_size=64)
      if patient_scans is None:
        continue
      feature = idx_to_feature_extractor[i](patient_scans, batching=True)
      feature = np.array([f.flatten() for f in feature])
      feature = np.mean(feature, axis=0).reshape(1, -1)
      feature = pcas[i].transform(feature)
      feature = feature.flatten()
      
      for j in range(len(x_train[patient_id])):
        x = x_train[patient_id][j]
        y = y_train[patient_id][j]

        features.append(np.array(feature.tolist()+list(x.values())))
        targets.append(y)
    rf = rf_regressors[i]
    rf.fit(features, targets)
  
  return rf_regressors

def get_dataset() -> tuple[defaultdict, defaultdict, defaultdict, defaultdict, defaultdict, defaultdict]:
  organize_dicom()

  # Prepare train and val data
  train_patients = pd.read_csv(data_dir + '/train.csv')
  test_patients = pd.read_csv(data_dir + '/test.csv')
  
  test_patients_ids = test_patients['Patient']

  # Remove row in train data if patient_id is in test_patient_ids
  test_data = train_patients[train_patients['Patient'].isin(test_patients_ids)]
  test_data = pd.concat([test_data, test_patients])
  test_data = test_data.sort_values(by=['Patient', 'Weeks'])
  test_patient_id_to_initial_FVC = test_data.groupby('Patient')['FVC'].first().to_dict()
  
  x_test = defaultdict(list)
  y_test = defaultdict(list)
  for row in test_data.itertuples():
    patient_id = row.Patient
    x_test[patient_id].append({
      'initial_FVC': test_patient_id_to_initial_FVC[patient_id],
      'week': row.Weeks
    })
    y_test[patient_id].append(row.FVC)
  
  train_patients = train_patients[~train_patients['Patient'].isin(test_patients_ids)]

  # Group by patient and get the first FVC value
  patient_id_to_initial_FVC = train_patients.groupby('Patient')['FVC'].first().to_dict()
  train_patients, val_patients = train_val_split(train_patients)

  # Sort df by patient and then by 'Weeks'
  train_patients = train_patients.sort_values(by=['Patient', 'Weeks'])

  x_train = defaultdict(list)
  y_train = defaultdict(list)

  for row in train_patients.itertuples():
    patient_id = row.Patient    
    x_train[patient_id].append({
      'initial_FVC': patient_id_to_initial_FVC[patient_id],
      'week': row.Weeks
    })
    y_train[patient_id].append(row.FVC)

  x_val = defaultdict(list)
  y_val = defaultdict(list)

  for row in val_patients.itertuples():
    patient_id = row.Patient    
    x_val[patient_id].append({
      'initial_FVC': patient_id_to_initial_FVC[patient_id],
      'week': row.Weeks
    })
    y_val[patient_id].append(row.FVC)
  
  return x_train, y_train, x_val, y_val, x_test, y_test

def initialize_models(x_train, y_train) -> tuple[list[PCA], list[RandomForestRegressor]]:
  if not os.path.exists(f'{checkpoints_dir}/pcas.pkl'):
    pcas = get_pcas(x_train)

    with open(f'{checkpoints_dir}/pcas.pkl', 'wb') as file:
      pickle.dump(pcas, file)
  else:
    with open(f'{checkpoints_dir}/pcas.pkl', 'rb') as file:
      pcas = pickle.load(file)

  if not os.path.exists(f'{checkpoints_dir}/rf_regressors.pkl'):
    rf_regressors = get_rf_regressors(pcas, x_train, y_train)
    with open(f'{checkpoints_dir}/rf_regressors.pkl', 'wb') as file:
      pickle.dump(rf_regressors, file)
  else:
    with open(f'{checkpoints_dir}/rf_regressors.pkl', 'rb') as file:
      rf_regressors = pickle.load(file)
  
  return pcas, rf_regressors

def get_predictions(pcas, rf_regressors, x):
  all_predictions = []
  for i in range(5):
    print(f"\nEvaluating {idx_to_feature_name[i]} model...")
    features = []
    targets = []
    
    for patient_id in tqdm(x.keys()):
      patient_scans = load_patient_scans(patient_id, scan_batch_size=64)
      if patient_scans is None:
        print(f"Could not load scans for {patient_id}")
        continue
      
      # Extract and transform features (same as training)
      feature = idx_to_feature_extractor[i](patient_scans, batching=True)
      feature = np.array([f.flatten() for f in feature])
      feature = np.mean(feature, axis=0).reshape(1, -1)
      feature = pcas[i].transform(feature)
      feature = feature.flatten()
      
      # Make predictions for each data point
      for j in range(len(x[patient_id])):
        point = x[patient_id][j]        
        features.append(np.array(feature.tolist() + list(point.values())))

    # Predict and compute MSE
    rf = rf_regressors[i]
    predictions = rf.predict(features)
    all_predictions.append(predictions)
  return all_predictions

if __name__ == "__main__":
  patient_scans = load_patient_scans('ID00009637202177434476278', scan_batch_size=64)
  for i in range(5):
    feature = idx_to_feature_extractor[0](patient_scans, batching=True)

