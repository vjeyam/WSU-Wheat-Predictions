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
    
    # Split the image into NIR and RGB
    nir = img[:, :640, :].astype(np.float64)
    rgb = img[:, 641:1280, :].astype(np.float64)
    
    # Extract a 5x5 region around the RGB center coordinates
    rgb_gray = img[rgb_y_center - 5:rgb_y_center + 5, rgb_x_center - 5:rgb_x_center + 5, :]
    if rgb_gray.size == 0:
        raise ValueError(f'RGB region extraction resulted in empty slice: {rgb_x_center}, {rgb_y_center}')
    
    # Extract the red, green, and blue bands from the gray region
    rgb_gray_red: np.ndarray = rgb_gray[:, :, 2].astype(np.float64)
    rgb_gray_green: np.ndarray = rgb_gray[:, :, 1].astype(np.float64)
    rgb_gray_blue: np.ndarray = rgb_gray[:, :, 0].astype(np.float64)
    rgb_gray_red_avg: float = np.mean(rgb_gray_red)
    rgb_gray_green_avg: float = np.mean(rgb_gray_green)
    rgb_gray_blue_avg: float = np.mean(rgb_gray_blue)
    
    # Convert the RGB image to red, green, blue bands
    red: np.ndarray = rgb[:, :, 2].astype(np.float64)
    green: np.ndarray = rgb[:, :, 1].astype(np.float64)
    blue: np.ndarray = rgb[:, :, 0].astype(np.float64)
    rgb_red_ref: np.ndarray = (red / rgb_gray_red_avg)
    rgb_green_ref: np.ndarray = (green / rgb_gray_green_avg)
    rgb_blue_ref: np.ndarray = (blue / rgb_gray_blue_avg)
    
    # Calculate the SCI (Soil Color Index)
    sci: np.ndarray = (rgb_red_ref - rgb_green_ref) / (rgb_red_ref + rgb_green_ref)
    
    # Extract a 5x5 region around the NIR center coordinates
    nir_gray = img[nir_y_center - 5:nir_y_center + 5, nir_x_center - 5:nir_x_center + 5, :]
    if nir_gray.size == 0:
        raise ValueError(f'NIR region extraction resulted in empty slice: {nir_x_center}, {nir_y_center}')
    
    # Extract the red, green, and blue bands from the gray region
    nir_gray_red: np.ndarray = nir_gray[:, :, 2].astype(np.float64)
    nir_gray_green: np.ndarray = nir_gray[:, :, 1].astype(np.float64)
    nir_gray_blue: np.ndarray = nir_gray[:, :, 0].astype(np.float64)
    nir_gray_red_avg: float = np.mean(nir_gray_red)
    nir_gray_green_avg: float = np.mean(nir_gray_green)
    nir_gray_blue_avg: float = np.mean(nir_gray_blue)
    
    # Convert the NIR image to red, green, blue bands
    nir_red: np.ndarray = nir[:, :, 2].astype(np.float64)
    nir_green: np.ndarray = nir[:, :, 1].astype(np.float64)
    nir_blue: np.ndarray = nir[:, :, 0].astype(np.float64)

    # Apply calibration factors if available
    if cam_name in calibration_data:
        calib_factors = calibration_data[cam_name]
        nir_red_ref: np.ndarray = nir_red / (nir_gray_red_avg * (calib_factors.get('Red', 1.0) / 100.0))
        nir_green_ref: np.ndarray = nir_green / (nir_gray_green_avg * (calib_factors.get('Green', 1.0) / 100.0))
        nir_blue_ref: np.ndarray = nir_blue / (nir_gray_blue_avg * (calib_factors.get('Blue', 1.0) / 100.0))
    else:
        print(f'No calibration data found for {cam_name}')
        nir_red_ref, nir_green_ref, nir_blue_ref = nir_red, nir_green, nir_blue
    
    # Check for zero denominator and handle division by zero
    denominator: np.ndarray = nir_red_ref + nir_green_ref
    if np.any(denominator == 0):
        print(f'Warning: Zero denominator found in GNDVI calculation.')
    denominator[denominator == 0] = np.nan  # Avoid division by zero by replacing 0 with NaN
    
    # Calculate the GNDVI
    gndvi: np.ndarray = (nir_red_ref - nir_green_ref) / denominator
    
    return sci, gndvi

def process_cameras(base_path='../data/', cam_prefix='cam'):
    for i in range(1, 9):
        cam_name = f'{cam_prefix}{i}'
        csv_file = f'../model_output/{cam_name}.csv'
        cam_folder = os.path.join(base_path, cam_name)
        
        # Read the CSV file to get the center coordinates
        df: pd.DataFrame = pd.read_csv(csv_file)
        
        # Separate NIR and RGB coordinates based on Center_X value
        nir_coords = df[df['Center_X'] > 640]
        rgb_coords = df[df['Center_X'] <= 640]
        
        # Assuming there's only one pair of coordinates for NIR and RGB in the CSV
        nir_x_center: float = float(nir_coords['Center_X'].values[0])
        nir_y_center: float = float(nir_coords['Center_Y'].values[0])
        rgb_x_center: float = float(rgb_coords['Center_X'].values[0])
        rgb_y_center: float = float(rgb_coords['Center_Y'].values[0])
        
        # Create directories if they don't exist
        sci_output_dir = os.path.join('../assets/vi/sci', cam_name)
        gndvi_output_dir = os.path.join('../assets/vi/gndvi', cam_name)
        os.makedirs(sci_output_dir, exist_ok=True)
        os.makedirs(gndvi_output_dir, exist_ok=True)
        
        # Process each image in the camera folder
        for image_file in os.listdir(cam_folder):
            if image_file.lower().endswith(('.png')):
                image_path = os.path.join(cam_folder, image_file)
                
                # Load the image
                image: np.ndarray = np.array(Image.open(image_path)).astype(np.float64)
                
                # Call the vi function with the appropriate parameters
                sci, gndvi = vi(image, nir_x_center, nir_y_center, rgb_x_center, rgb_y_center, cam_name)
                print(f'GNDVI for {cam_name}, {image_file}: {np.nanmean(gndvi)}')  # Print mean value for debugging
                
                # Save the results in the assets folder
                base_filename = os.path.splitext(image_file)[0]
                
                # Save SCI and GNDVI images as grayscale
                sci_image = Image.fromarray((sci * 255).astype(np.uint8))
                gndvi_image = Image.fromarray((gndvi * 255).astype(np.uint8))
                
                sci_image.save(os.path.join(sci_output_dir, f'{base_filename}_sci.png'))
                gndvi_image.save(os.path.join(gndvi_output_dir, f'{base_filename}_gndvi.png'))

if __name__ == "__main__":
    process_cameras()