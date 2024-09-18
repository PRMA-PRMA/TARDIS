# filtering_modifications.py
import numpy as np
from skimage import restoration
from scipy.ndimage import gaussian_filter, median_filter

def apply_gaussian_filter(img_data, sigma=1):
    """Apply Gaussian filter to the image."""
    return gaussian_filter(img_data, sigma=sigma)

def apply_median_filter(img_data, size=3):
    """Apply Median filter to the image."""
    return median_filter(img_data, size=size)

def apply_non_local_means(img_data, patch_size=5, patch_distance=6, h=0.1):
    """Apply Non-Local Means denoising to the image."""
    return restoration.denoise_nl_means(
        img_data,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=True,
        multichannel=False
    )
