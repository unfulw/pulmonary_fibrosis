import pydicom
import numpy as np
import cv2
from glob import glob
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import nibabel as nib

# reads a single dicom image and converts it to a numpy array
def load_dicom_image(file_path):
    """Load a DICOM image and return it as a numpy array."""
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array
    return image

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess the image: resize and normalize."""
    image_resized = cv2.resize(image, target_size)
    image_normalized = (image_resized - np.min(image_resized)) / (np.max(image_resized) - np.min(image_resized))
    return image_normalized

# reads all dicom images from a directory and preprocesses them
def load_dicom_images_from_directory(directory):
    """Load and preprocess all DICOM images from a directory."""
    file_paths = glob(os.path.join(directory, '*.dcm'))
    images = []
    for file_path in file_paths:
        image = load_dicom_image(file_path)
        image_preprocessed = preprocess_image(image)
        images.append(image_preprocessed)
    return np.array(images)

def dicom_to_hu(path: str) -> np.ndarray:
    """Convert DICOM pixel data to Hounsfield Units (HU)."""
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array.astype(np.int16)

    # Convert to HU
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope 

    image = slope * image + intercept

    return image

def lung_window(img_hu: np.ndarray, center = -600, width = 1500) -> np.ndarray:
    """Apply lung windowing to the HU image."""

    min_hu = center - (width // 2)
    max_hu = center + (width // 2)

    img_windowed = np.clip(img_hu, min_hu, max_hu) 

    return img_windowed

def load_patient_stack(dicom_dir: str, target_depth: int = 96, img_size: int = 224) -> np.ndarray:
    """Load all slices, sort by InstanceNumber, window, resize, depth-pad/trim."""
    files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOMs in {dicom_dir}")

    # sort slices using InstanceNumber or filename fallback
    def inst_no(fp):
        try:
            return int(pydicom.dcmread(fp, stop_before_pixels=True).InstanceNumber)
        except Exception:
            return 0
    files.sort(key=inst_no)

    slices = [lung_window(dicom_to_hu(fp)) for fp in files]

    # resize each slice
    slices = [cv2.resize(s, (img_size, img_size), interpolation=cv2.INTER_AREA) for s in slices]
    vol = np.stack(slices, axis=0)  # [D, H, W]

    # normalize per-volume (optional but helpful)
    v = vol.astype(np.float32)
    v = (v - v.mean()) / (v.std() + 1e-6)

    # depth pad/trim to target_depth
    D = v.shape[0]
    if D == target_depth:
        pass
    elif D > target_depth:
        # uniform downsample to target_depth
        idx = np.linspace(0, D-1, target_depth).round().astype(int)
        v = v[idx]
    else:
        # pad by symmetric reflection
        pad_needed = target_depth - D
        pre = pad_needed // 2
        post = pad_needed - pre
        v = np.pad(v, ((pre, post), (0,0), (0,0)), mode="reflect")

    # channel-first for 2D CNN over slices: keep as [D, H, W]; we’ll add channel later
    return v  # float32

import nibabel as nib
ct_image = nib.load(os.path)

dir = "/Users/rlaal/Documents/GitHub/pulmonary_fibrosis/data"
train_csv = pd.read_csv(os.path.join(dir, 'train.csv'))
train_path = os.path.join(dir, 'train')

train_path = os.path.join(dir, 'train')
test_path = os.path.join(dir, 'test')

train_patients = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
test_patients = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]

print(len(train_patients), len(test_patients))

# Number of patients in the training tabular file 
train_csv = pd.read_csv(os.path.join(dir, 'train.csv'))
print(len(train_csv)) # prints the number of rows in the CSV file

# Number of patients in the training ct scan folder
train_patients = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
print(len(train_patients)) # prints the number of patient folders in the training folder

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(224,224,1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)
    return models.Model(inputs, outputs)

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd


X_train = load_dicom_images_from_directory(os.path.join(train_path, patient_id))
y_train = train_csv[train_csv['Patient'] == patient_id]['FVC'].values

X_test = load_dicom_images_from_directory(os.path.join(test_path, patient_id))
y_test = train_csv[train_csv['Patient'] == patient_id]['FVC'].values

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='linear')  # regression for FVC
])

model.compile(optimizer='adam', loss='mae', metrics=['mse'])

# Suppose X_train and y_train are ready
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

