import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Tuple, Dict

def process_rgb_images(input_folder, output_folder_rgb) -> None:
    """
    Process RGB images by splitting them into two halves and saving the right half.
    The function assumes that the input images are in folders named 'cam1' to 'cam8'.
    Each camera folder contains images with the same naming convention.
    The right half of each image is saved in a corresponding output folder named 'cam1_rgb' to 'cam8_rgb'.
    The function checks the image size to ensure it is 1280 x 928 before processing.
    If the image size is not as expected, it skips the image and prints a message.
    The function creates the output folders if they do not exist.

    Args:
        input_folder (_type_): _description_
        output_folder_rgb (_type_): _description_
    
    Returns:
        None
    """
    # List all camera folders
    cam_folders = [f'cam{i}' for i in range(1, 9)]

    for cam in cam_folders:
        cam_path = os.path.join(input_folder, cam)
        output_cam_rgb_path = os.path.join(output_folder_rgb, f'{cam}_rgb')
        os.makedirs(output_cam_rgb_path, exist_ok=True)
        
        # List all image files in the current camera folder
        for filename in os.listdir(cam_path):
            if filename.endswith('.png'):
                image_path = os.path.join(cam_path, filename)
                img = Image.open(image_path)
                
                # Check the image size to ensure it is 1280 x 928
                if img.size != (1280, 928):
                    print(f"Skipping {image_path}: unexpected image size {img.size}")
                    continue
                
                # Split the image to get the right half and save it
                right_half = img.crop((640, 0, 1280, 928))
                right_half.save(os.path.join(output_cam_rgb_path, f'{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}'))
                print(f"Processed right side of {filename}")

def process_nir_images(input_folder, output_folder_ir) -> None:
    """
    Process NIR images by splitting them into two halves and saving the left half.
    The function assumes that the input images are in folders named 'cam1' to 'cam8'.
    Each camera folder contains images with the same naming convention.
    The left half of each image is saved in a corresponding output folder named 'cam1_nir' to 'cam8_nir'.
    The function checks the image size to ensure it is 1280 x 928 before processing.
    If the image size is not as expected, it skips the image and prints a message.
    The function creates the output folders if they do not exist.

    Args:
        input_folder (_type_): _description_
        output_folder_ir (_type_): _description_
        
    Returns:
        None
    """
    # List all camera folders
    cam_folders = [f'cam{i}' for i in range(1, 9)]

    for cam in cam_folders:
        cam_path = os.path.join(input_folder, cam)
        output_cam_ir_path = os.path.join(output_folder_ir, f'{cam}_nir')   
        os.makedirs(output_cam_ir_path, exist_ok=True)
        
        # List all image files in the current camera folder
        for filename in os.listdir(cam_path):
            if filename.endswith('.png'):
                image_path = os.path.join(cam_path, filename)
                img = Image.open(image_path)
                
                # Check the image size to ensure it is 1280 x 928
                if img.size != (1280, 928):
                    print(f"Skipping {image_path}: unexpected image size {img.size}")
                    continue
                
                # Split the image to get the left half and save it
                left_half = img.crop((0, 0, 640, 928))
                left_half.save(os.path.join(output_cam_ir_path, f'{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}'))
                print(f"Processed left side of {filename}")

def reorder_csv_files(model_output_dir: str) -> None:
    """
    Reorder CSV files in the model output directory by removing rows with NaN values in 'Center_X' and 'Center_Y' columns,
    and sorting the DataFrame by 'Filename' column.
    The function processes files with prefixes 'cam1' to 'cam8' and suffixes '_nir.csv' and '_rgb.csv'.
    The processed DataFrames are saved back to the same CSV files.
    The function prints a message for each processed file.
    The function creates the output folders if they do not exist

    Args:
        model_output_dir (str): _description_
    
    Returns:
        None
    """
    # Define the prefixes and suffixes for the CSV files to be processed
    prefixes = [f'cam{i}' for i in range(1, 9)]
    suffixes = ['_nir.csv', '_rgb.csv']
    all_files = os.listdir(model_output_dir)
    csv_files = [f for f in all_files if any(f.startswith(prefix) and f.endswith(suffix) for prefix in prefixes for suffix in suffixes)]
    
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(model_output_dir, csv_file))
        df = df.dropna(subset=['Center_X', 'Center_Y']).sort_values(by='Filename')
        df.to_csv(os.path.join(model_output_dir, csv_file), index=False)
        print(f"Processed {csv_file}")

def plot_center_coordinates(model_output_dir: str, assets_dir: str) -> None:
    for i in range(1, 9):
        for mode in ['nir', 'rgb']:
            df = pd.read_csv(f'{model_output_dir}/cam{i}_{mode}.csv')
            plt.figure(figsize=(10, 7))
            plt.scatter(df['Center_X'], df['Center_Y'], c='r' if mode == 'nir' else 'b', marker='o')
            plt.title(f'Camera {i} {mode.upper()} Center_X and Center_Y values')
            plt.xlabel('Center_X')
            plt.ylabel('Center_Y')
            plt.grid(True)
            os.makedirs(assets_dir, exist_ok=True)
            plt.savefig(f'{assets_dir}/cam{i}_{mode}_center_x_y.png')
            plt.close()

def calculate_panel_rgb_values(csv_file: str, image_folder: str) -> pd.DataFrame:
    """
    Calculate the average RGB values for a 5x5 region around the center coordinates in the images.
    The function reads a CSV file containing the center coordinates and filenames of the images,
    processes each image to extract the 5x5 region around the center coordinates,
    and calculates the average RGB values for that region.
    The function returns a DataFrame containing the average RGB values for each image.
    
    Args:
        csv_file (str): _description_
        image_folder (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Load the CSV file and process each image
    data = pd.read_csv(csv_file)
    results = []
    
    for _, row in data.iterrows():
        image_path = os.path.join(image_folder, row['Filename'])
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Extract and 5x5 region around the center coordinates
        x_start, x_end = max(0, int(row['Center_X']) - 5), min(image.shape[1], int(row['Center_X']) + 5)
        y_start, y_end = max(0, int(row['Center_Y']) - 5), min(image.shape[0], int(row['Center_Y']) + 5)
        region = image[y_start:y_end, x_start:x_end]
        avg_color = region.mean(axis=(0, 1))
        results.append({
            'Filename': row['Filename'],
            'Center_X': row['Center_X'],
            'Center_Y': row['Center_Y'],
            'Average_R': avg_color[2],
            'Average_G': avg_color[1],
            'Average_B': avg_color[0]
        })
    return pd.DataFrame(results)

calibration_data: Dict[str, Dict[str, float]] = {
    f'cam{i}': {'Blue': 10+i, 'Red': 12+i, 'Green': 12+i} for i in range(1, 9)
}

def vi(img: np.ndarray, nir_x: float, nir_y: float, rgb_x: float, rgb_y: float, cam_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Soil Chlorophyll Index (SCI) and Green Normalized Difference Vegetation Index (GNDVI) for the given image.
    The function extracts the NIR and RGB channels from the image, computes the average values for a small region around the specified coordinates,
    and then calculates the indices based on the average values and calibration data.
    The function returns the SCI and GNDVI as numpy arrays.
    The function assumes that the input image is in the format of (height, width, channels) and that the NIR channel is in the first 640 columns`
    and the RGB channel is in the next 640 columns.

    Args:
        img (np.ndarray): _description_
        nir_x (float): _description_
        nir_y (float): _description_
        rgb_x (float): _description_
        rgb_y (float): _description_
        cam_name (str): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    height, width, _ = img.shape
    rgb_x, rgb_y, nir_x, nir_y = map(lambda v: int(round(v)), (rgb_x, rgb_y, nir_x, nir_y))
    nir = img[:, :640, :].astype(np.float64)
    rgb = img[:, 641:1280, :].astype(np.float64)
    rgb_region = img[rgb_y-5:rgb_y+5, rgb_x-5:rgb_x+5, :]
    nir_region = img[nir_y-5:nir_y+5, nir_x-5:nir_x+5, :]
    rgb_avg = np.mean(rgb_region, axis=(0, 1))
    nir_avg = np.mean(nir_region, axis=(0, 1))
    red, green = rgb[:, :, 2], rgb[:, :, 1]
    sci = (red/rgb_avg[2] - green/rgb_avg[1]) / (red/rgb_avg[2] + green/rgb_avg[1])
    nir_red, nir_green = nir[:, :, 2], nir[:, :, 1]
    if cam_name in calibration_data:
        calib = calibration_data[cam_name]
        nir_red /= (nir_avg[2] * calib['Red']/100)
        nir_green /= (nir_avg[1] * calib['Green']/100)
    gndvi = (nir_red - nir_green) / (nir_red + nir_green)
    return sci, gndvi

def process_cameras(data_dir: str, model_output_dir: str, assets_dir: str) -> None:
    """
    Process the camera images to calculate the Soil Chlorophyll Index (SCI) and Green Normalized Difference Vegetation Index (GNDVI).
    The function reads the CSV files containing the center coordinates for each camera,
    extracts the NIR and RGB coordinates, and processes the images to calculate the indices.

    Args:
        data_dir (str): _description_
        model_output_dir (str): _description_
        assets_dir (str): _description_
    """
    for i in range(1, 9):
        cam_name = f'cam{i}'
        df = pd.read_csv(f'{model_output_dir}/{cam_name}.csv')
        nir_x, nir_y = df[df['Center_X'] > 640][['Center_X', 'Center_Y']].values[0]
        rgb_x, rgb_y = df[df['Center_X'] <= 640][['Center_X', 'Center_Y']].values[0]
        cam_folder = os.path.join(data_dir, cam_name)
        for img_file in os.listdir(cam_folder):
            if img_file.lower().endswith('.png'):
                img = np.array(Image.open(os.path.join(cam_folder, img_file))).astype(np.float64)
                sci, gndvi = vi(img, nir_x, nir_y, rgb_x, rgb_y, cam_name)
                base_name = os.path.splitext(img_file)[0]
                os.makedirs(f'{assets_dir}/vi/sci/{cam_name}', exist_ok=True)
                os.makedirs(f'{assets_dir}/vi/gndvi/{cam_name}', exist_ok=True)
                Image.fromarray((sci*255).astype(np.uint8)).save(f'{assets_dir}/vi/sci/{cam_name}/{base_name}_sci.png')
                Image.fromarray((gndvi*255).astype(np.uint8)).save(f'{assets_dir}/vi/gndvi/{cam_name}/{base_name}_gndvi.png')

def main():
    input_folder = '../data'
    model_output_dir = '../model_output'
    assets_dir = '../assets'
    process_nir_images(input_folder, input_folder)
    reorder_csv_files(model_output_dir)
    plot_center_coordinates(model_output_dir, assets_dir)
    for i in range(1, 9):
        csv_path = f"{model_output_dir}/cam{i}_centers.csv"
        image_folder = f"{model_output_dir}/cam{i}"
        calculate_panel_rgb_values(csv_path, image_folder)
    process_cameras(input_folder, model_output_dir, assets_dir)

if __name__ == "__main__":
    main()
