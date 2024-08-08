import numpy as np
import pandas as pd
import os
from PIL import Image
from typing import Tuple, Dict

# Calibration data for 8 cameras
calibration_data: Dict[str, Dict[str, float]] = {
    'cam1': {'Blue': 10.77, 'Red': 12.24, 'Green': 12.57},
    'cam2': {'Blue': 12.18, 'Red': 11.67, 'Green': 13.82},
    'cam3': {'Blue': 11.58, 'Red': 11.35, 'Green': 12.63},
    'cam4': {'Blue': 11.27, 'Red': 10.99, 'Green': 12.49},
    'cam5': {'Blue': 11.22, 'Red': 12.59, 'Green': 14.87},
    'cam6': {'Blue': 11.13, 'Red': 11.10, 'Green': 12.04},
    'cam7': {'Blue': 11.36, 'Red': 11.35, 'Green': 12.53},
    'cam8': {'Blue': 11.00, 'Red': 12.27, 'Green': 14.01}
}

def vi(img: np.ndarray, 
       nir_x_center: float, 
       nir_y_center: float, 
       rgb_x_center: float, 
       rgb_y_center: float, 
       cam_name: str
       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Soil Color Index (SCI) and Green Normalized Difference Vegetation Index (GNDVI)
    for a given image based on NIR and RGB center coordinates.

    Args:
        img (np.ndarray): The input image array.
        nir_x_center (float): X-coordinate of the NIR center.
        nir_y_center (float): Y-coordinate of the NIR center.
        rgb_x_center (float): X-coordinate of the RGB center.
        rgb_y_center (float): Y-coordinate of the RGB center.
        cam_name (str): Name of the camera to apply calibration data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: SCI and GNDVI arrays.
    """
    height, width, _ = img.shape
    
    # Ensure coordinates are within bounds and cast to integers
    rgb_x_center = int(round(rgb_x_center))
    rgb_y_center = int(round(rgb_y_center))
    nir_x_center = int(round(nir_x_center))
    nir_y_center = int(round(nir_y_center))
    
    if not (5 <= rgb_x_center < width - 5 and 5 <= rgb_y_center < height - 5):
        raise ValueError(f'RGB coordinates out of bounds: {rgb_x_center}, {rgb_y_center}')
    if not (5 <= nir_x_center < width - 5 and 5 <= nir_y_center < height - 5):
        raise ValueError(f'NIR coordinates out of bounds: {nir_x_center}, {nir_y_center}')

    # SCI: Red divided by Blue
    sci = np.divide(img[rgb_y_center, rgb_x_center, 2], img[rgb_y_center, rgb_x_center, 0], out=np.zeros_like(img[..., 0]), where=img[..., 0] != 0)
    
    # GNDVI: (NIR - Green) / (NIR + Green)
    nir = img[nir_y_center, nir_x_center, 0]
    green = img[rgb_y_center, rgb_x_center, 1]
    gndvi = np.divide((nir - green), (nir + green), out=np.zeros_like(nir), where=(nir + green) != 0)

    return sci, gndvi

def radiometric_correction(img: np.ndarray, 
                           cam_name: str, 
                           lut_values: Dict[str, float]
                           ) -> np.ndarray:
    """
    Applies radiometric correction to an image based on camera calibration data.

    Args:
        img (np.ndarray): The input image array.
        cam_name (str): Name of the camera.
        lut_values (Dict[str, float]): Lookup table values for correction.

    Returns:
        np.ndarray: The radiometrically corrected image.
    """
        # Apply correction using calibration data
    img_corrected = np.zeros_like(img, dtype=np.float32)
    
    img_corrected[..., 0] = img[..., 0] / lut_values['Blue']
    img_corrected[..., 1] = img[..., 1] / lut_values['Green']
    img_corrected[..., 2] = img[..., 2] / lut_values['Red']
    
    # Clip values to ensure they are within the correct range
    np.clip(img_corrected, 0, 255, out=img_corrected)
    
    return img_corrected.astype(np.uint8)

def apply_correction_to_all_images(input_dir: str, output_dir: str) -> None:
    """
    Applies radiometric correction to all images in the input directory
    and saves the corrected images to the output directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where corrected images will be saved.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for cam_name in calibration_data.keys():
        cam_input_dir = os.path.join(input_dir, cam_name)
        cam_output_dir = os.path.join(output_dir, cam_name)
        
        if not os.path.exists(cam_output_dir):
            os.makedirs(cam_output_dir)
        
        lut_values = calibration_data[cam_name]
        
        for filename in os.listdir(cam_input_dir):
            img_path = os.path.join(cam_input_dir, filename)
            img = np.array(Image.open(img_path))
            
            # Apply radiometric correction
            img_corrected = radiometric_correction(img, cam_name, lut_values)
            
            # Save the corrected image
            output_path = os.path.join(cam_output_dir, filename)
            Image.fromarray(img_corrected).save(output_path)
            print(f"Saved corrected image: {output_path}")

if __name__ == "__main__":
    input_dir = '../datasets/wheat/train/images'
    output_dir = '../datasets/wheat/train/corrected_images'
    
    # Apply radiometric corrections to all images in the dataset
    apply_correction_to_all_images(input_dir, output_dir)