import numpy as np
import pydicom
import scipy.ndimage as ndimage
import skimage.morphology as morphology

def rescale_to_hu(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    # Ensure dicom file is in Hounsfield unit by rescaling
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    hu_image = dcm.pixel_array * slope + intercept
    return hu_image

def window_image(image, window_center, window_width):
    # CT Windowing, min = window_center - window_width // 2, max = window_center + window_width // 2
    # Clip values outside of window to img_min or img_max
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image

def get_mask(image: np.ndarray, window_level: int = 40, window_width: int = 80):
    # Reference wl and ww values: https://radiopaedia.org/articles/windowing-ct
    # Default value for bone tissue

    thresholded_image = window_image(image, window_level, window_width)

    segmentation = morphology.dilation(thresholded_image, np.ones((4, 4)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0

    mask = labels == label_count.argmax()

    # Improve the mask
    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))

    return mask

def preprocess_dicom(dcm: pydicom.dataset.FileDataset) -> np.ndarray:

  hu_scan = rescale_to_hu(dcm)
  mask = get_mask(hu_scan)
  preprocessed_scan = np.float32(hu_scan * mask)

  return preprocessed_scan